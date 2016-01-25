/*
 *  This file is a part of Libint.
 *  Copyright (C) 2004-2014 Edward F. Valeev
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 */

// standard C++ headers
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <boost/functional/hash.hpp>
#include <mutex>

// Eigen matrix algebra library
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>

// have BTAS library?
#ifdef LIBINT2_HAVE_BTAS
# include <btas/btas.h>
#endif // LIBINT2_HAVE_BTAS

// Libint Gaussian integrals library
#include <libint2.hpp>
#include <libint2/diis.h>

#if defined(_OPENMP)
# include <omp.h>
#endif

// uncomment if want to report integral timings (only useful if nthreads == 1)
// N.B. integral engine timings are controled in engine.h
//#define REPORT_INTEGRAL_TIMINGS
//
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        MatrixI;  

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        Matrix;  // import dense, dynamically sized Matrix type from Eigen;
                 // this is a matrix with row-major storage (http://en.wikipedia.org/wiki/Row-major_order)
                 // to meet the layout of the integrals returned by the Libint integral library
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic>
        DiagonalMatrix;

using libint2::Shell;
using libint2::Atom;
using libint2::BasisSet;

std::vector<Atom> read_geometry(const std::string& filename);
Matrix compute_soad(const std::vector<Atom>& atoms);
// computes norm of shell-blocks of A
Matrix compute_shellblock_norm(const BasisSet& obs,
                               const Matrix& A);

template <libint2::OneBodyEngine::operator_type obtype>
std::array<Matrix, libint2::OneBodyEngine::operator_traits<obtype>::nopers>
compute_1body_ints(const BasisSet& obs,
                   const std::vector<Atom>& atoms = std::vector<Atom>());

#if LIBINT2_DERIV_ONEBODY_ORDER
template <libint2::OneBodyEngine::operator_type obtype>
std::vector<Matrix>
compute_1body_deriv_ints(unsigned deriv_order,
                         const BasisSet& obs,
                         const std::vector<Atom>& atoms);
#endif

Matrix compute_schwartz_ints(const BasisSet& bs1,
                             const BasisSet& bs2 = BasisSet(),
                             bool use_2norm = false // use infty norm by default
                            );
Matrix compute_do_ints(const BasisSet& bs1,
                       const BasisSet& bs2 = BasisSet(),
                       bool use_2norm = false // use infty norm by default
                      );

using shellpair_list_t = std::unordered_map<size_t,std::vector<size_t>>;
shellpair_list_t obs_shellpair_list; // shellpair list for OBS

std::vector<double> k_times;
std::vector<int> j_ints;
std::vector<int> k_ints;

/// computes non-negligible shell pair list; shells \c i and \c j form a non-negligible
/// pair if they share a center or the Frobenius norm of their overlap is greater than threshold
shellpair_list_t
compute_shellpair_list(const BasisSet& bs1,
                       const BasisSet& bs2 = BasisSet(),
                       double threshold = std::numeric_limits<double>::epsilon()
                      );

Matrix compute_2body_fock_fake(const BasisSet& obs,
                          const Matrix& D,
                          double precision,
                          const Matrix& Schwartz,
                          bool use_linK = false);

Matrix compute_2body_fock(const BasisSet& obs,
                          const Matrix& D,
                          double j_precision = std::numeric_limits<double>::epsilon(), // discard contributions smaller than this
                          double k_precision = std::numeric_limits<double>::epsilon(), // discard contributions smaller than this
                          const Matrix& Schwartz = Matrix(), // K_ij = sqrt(||(ij|ij)||_\infty); if empty, do not Schwartz screen
                          const bool use_linK = false
                         );

// an Fock builder that can accept densities expressed a separate basis
Matrix compute_2body_fock_general(const BasisSet& obs,
                                  const Matrix& D,
                                  const BasisSet& D_bs,
                                  bool D_is_sheldiagonal = false, // set D_is_shelldiagonal if doing SOAD
                                  double precision = std::numeric_limits<double>::epsilon() // discard contributions smaller than this
                                 );

#ifdef LIBINT2_HAVE_BTAS
# define HAVE_DENSITY_FITTING 1
  struct DFFockEngine {
    const BasisSet& obs;
    const BasisSet& dfbs;
    DFFockEngine(const BasisSet& _obs, const BasisSet& _dfbs) :
      obs(_obs), dfbs(_dfbs)
    {
    }

    typedef btas::RangeNd<CblasRowMajor, std::array<long, 3> > Range3d;
    typedef btas::Tensor<double, Range3d> Tensor3d;
    Tensor3d xyK;

    // a DF-based builder, using coefficients of occupied MOs
    Matrix compute_2body_fock_dfC(const Matrix& Cocc);
  };
#endif // HAVE_DENSITY_FITTING

namespace libint2 {
  int nthreads;

  /// fires off \c nthreads instances of lambda in parallel
  template <typename Lambda>
  void parallel_do(Lambda& lambda) {
#ifdef _OPENMP
  #pragma omp parallel
  {
    auto thread_id = omp_get_thread_num();
    lambda(thread_id);
  }
#else // use C++11 threads
  std::vector<std::thread> threads;
  for(int thread_id=0; thread_id != libint2::nthreads; ++thread_id) {
    if(thread_id != nthreads-1)
      threads.push_back(std::thread(lambda,
                                    thread_id));
    else
      lambda(thread_id);
  } // threads_id
  for(int thread_id=0; thread_id<nthreads-1; ++thread_id)
    threads[thread_id].join();
#endif
  }

}

int main(int argc, char *argv[]) {

  using std::cout;
  using std::cerr;
  using std::endl;

  try {

    /*** =========================== ***/
    /*** initialize molecule         ***/
    /*** =========================== ***/

    // read geometry from a file; by default read from h2o.xyz, else take filename (.xyz) from the command line
    const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";
    const auto basisname = (argc > 2) ? argv[2] : "aug-cc-pVDZ";
    auto j_precision = (argc > 3) ? std::stod(argv[3]) : 1e-14;
    j_precision = std::max(j_precision, std::numeric_limits<double>::epsilon());

    auto k_precision = (argc > 4) ? std::stod(argv[4]) : 1e-14;
    k_precision = std::max(k_precision, std::numeric_limits<double>::epsilon());

    const bool use_linK = (argc > 5) ? std::stoi(argv[5]) : false;

    if(use_linK){
        std::cout << "Using LinK for exchange!" << std::endl;
    }

    bool do_density_fitting = false;
    std::vector<Atom> atoms = read_geometry(filename);

    // set up thread pool
    {
      using libint2::nthreads;
      auto nthreads_cstr = getenv("LIBINT_NUM_THREADS");
      nthreads = 1;
      if (nthreads_cstr && strcmp(nthreads_cstr,"")) {
        std::istringstream iss(nthreads_cstr);
        iss >> nthreads;
        if (nthreads > 1<<16 || nthreads <= 0)
          nthreads = 1;
      }
#if defined(_OPENMP)
      omp_set_num_threads(nthreads);
#endif
      std::cout << "Will scale over " << nthreads
#if defined(_OPENMP)
                << " OpenMP"
#else
                << " C++11"
#endif
                << " threads" << std::endl;
    }

    // count the number of electrons
    auto nelectron = 0;
    for (auto i = 0; i < atoms.size(); ++i)
      nelectron += atoms[i].atomic_number;
    const auto ndocc = nelectron / 2;
    cout << "# of electrons = " << nelectron << endl;

    // compute the nuclear repulsion energy
    auto enuc = 0.0;
    for (auto i = 0; i < atoms.size(); i++)
      for (auto j = i + 1; j < atoms.size(); j++) {
        auto xij = atoms[i].x - atoms[j].x;
        auto yij = atoms[i].y - atoms[j].y;
        auto zij = atoms[i].z - atoms[j].z;
        auto r2 = xij*xij + yij*yij + zij*zij;
        auto r = sqrt(r2);
        enuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
      }
    cout << "Nuclear repulsion energy = " << std::setprecision(15) << enuc << endl;

    libint2::Shell::do_enforce_unit_normalization(false);

    // cout << "Atomic Cartesian coordinates (a.u.):" << endl;
    // for(const auto& a: atoms)
    //   std::cout << a.atomic_number << " " << a.x << " " << a.y << " " << a.z << std::endl;

    BasisSet obs(basisname, atoms);
    cout << "orbital basis set rank = " << obs.nbf() << endl;

#ifdef HAVE_DENSITY_FITTING
    BasisSet dfbs;
    if (do_density_fitting) {
      dfbs = BasisSet(dfbasisname, atoms);
      cout << "density-fitting basis set rank = " << dfbs.nbf() << endl;
    }
    DFFockEngine dffockengine(obs,dfbs);
#endif // HAVE_DENSITY_FITTING


    /*** =========================== ***/
    /*** compute 1-e integrals       ***/
    /*** =========================== ***/

    // initializes the Libint integrals library ... now ready to compute
    libint2::init();

    // compute OBS non-negligible shell-pair list
    {
      obs_shellpair_list = compute_shellpair_list(obs);
      size_t nsp = 0;
      for(auto& sp: obs_shellpair_list) {
        nsp += sp.second.size();
      }
      std::cout << "# of {all,non-negligible} shell-pairs = {"
                << obs.size()*(obs.size()+1)/2 << ","
                << nsp << "}" << std::endl;
    }

    // compute one-body integrals
    auto S = compute_1body_ints<libint2::OneBodyEngine::overlap>(obs)[0];
    auto T = compute_1body_ints<libint2::OneBodyEngine::kinetic>(obs)[0];
    auto V = compute_1body_ints<libint2::OneBodyEngine::nuclear>(obs, atoms)[0];
    Matrix H = T + V;
    T.resize(0,0);
    V.resize(0,0);

    Matrix D;
    Matrix C_occ;
    Matrix evals;
    {  // use SOAD as the guess density
      const auto tstart = std::chrono::high_resolution_clock::now();

      auto D_minbs = compute_soad(atoms); // compute guess in minimal basis
      BasisSet minbs("STO-3G", atoms);
      if (minbs == obs)
        D = D_minbs;
      else { // if basis != minimal basis, map non-representable SOAD guess into the AO basis
             // by diagonalizing a Fock matrix
        std::cout << "projecting SOAD into AO basis ... ";
        auto F = H;
        F += compute_2body_fock_general(obs, D_minbs, minbs,
                                        true /* SOAD_D_is_shelldiagonal */,
                                        1e-12 // this is cheap, no reason to be cheaper
                                       );

        // solve F C = e S C
        Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(F, S);
        auto C = gen_eig_solver.eigenvectors();

        // compute density, D = C(occ) . C(occ)T
        C_occ = C.leftCols(ndocc);
        D = C_occ * C_occ.transpose();

        const auto tstop = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> time_elapsed = tstop - tstart;
        std::cout << "done (" << time_elapsed.count() << " s)" << std::endl;

      }
    }

    // pre-compute data for Schwartz bounds
    auto K = compute_schwartz_ints(obs);

    /*** =========================== ***/
    /***          SCF loop           ***/
    /*** =========================== ***/

    const auto maxiter = 20;
    const auto conv = 1e-12;
    auto iter = 0;
    auto rms_error = 1.0;
    auto ediff_rel = 0.0;
    auto ehf = 0.0;
    auto n2 = D.cols() * D.rows();
    libint2::DIIS<Matrix> diis(2); // start DIIS on second iteration

    std::cout << "J precision = " << j_precision << std::endl;
    std::cout << "K precision = " << k_precision << std::endl;

    // prepare for incremental Fock build ...
    Matrix D_diff = D;
    Matrix F = H;
    bool reset_incremental_fock_formation = false;
    bool incremental_Fbuild_started = false;
    double start_incremental_F_threshold = 1e-17;
    double next_reset_threshold = 0.0;
    size_t last_reset_iteration = 0;
    // ... unless doing DF, then use MO coefficients, hence not "incremental"
    if (do_density_fitting) start_incremental_F_threshold = 0.0;

    do {
      const auto tstart = std::chrono::high_resolution_clock::now();
      ++iter;

      // Last iteration's energy and density
      auto ehf_last = ehf;
      Matrix D_last = D;

      if (not incremental_Fbuild_started && rms_error < start_incremental_F_threshold) {
        incremental_Fbuild_started = true;
        reset_incremental_fock_formation = false;
        last_reset_iteration = iter - 1;
        next_reset_threshold = rms_error / 1e1;
        std::cout << "== started incremental fock build" << std::endl;
      }
      if (reset_incremental_fock_formation || not incremental_Fbuild_started) {
        F = H;
        D_diff = D;
      }
      if (reset_incremental_fock_formation && incremental_Fbuild_started) {
          reset_incremental_fock_formation = false;
          last_reset_iteration = iter;
          next_reset_threshold = rms_error / 1e1;
          std::cout << "== reset incremental fock build" << std::endl;
      }

      // build a new Fock matrix
      if (not do_density_fitting) {
        F += compute_2body_fock(obs, D_diff, j_precision, k_precision, K,
                use_linK);
      }
#if HAVE_DENSITY_FITTING
      else { // do DF
        F = H + dffockengine.compute_2body_fock_dfC(C_occ);
      }
#else
      else { assert(false); } // do_density_fitting is true but HAVE_DENSITY_FITTING is not defined! should not happen
#endif // HAVE_DENSITY_FITTING

      // compute HF energy with the non-extrapolated Fock matrix
      ehf = D.cwiseProduct(H+F).sum();
      ediff_rel = std::abs((ehf - ehf_last)/ehf);

      // compute SCF error
      Matrix FD_comm = F*D*S - S*D*F;
      rms_error = FD_comm.norm()/n2;
      if (rms_error < next_reset_threshold || iter - last_reset_iteration >= 8)
        reset_incremental_fock_formation = true;

      // DIIS extrapolate F
      Matrix F_diis = F; // extrapolated F cannot be used in incremental Fock build; only used to produce the density
                         // make a copy of the unextrapolated matrix
      diis.extrapolate(F_diis,FD_comm);

      // solve F C = e S C
      Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> gen_eig_solver(F_diis, S);
      evals = gen_eig_solver.eigenvalues();
      auto C = gen_eig_solver.eigenvectors();

      // compute density, D = C(occ) . C(occ)T
      C_occ = C.leftCols(ndocc);
      D = C_occ * C_occ.transpose();
      D_diff = D - D_last;

      const auto tstop = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> time_elapsed = tstop - tstart;

      if (iter == 1)
        std::cout <<
        "\n\nIter         E(HF)                 D(E)/E         RMS([F,D])/nn       Time(s)\n";
      printf(" %02d %20.12f %20.12e %20.12e %10.5lf\n", iter, ehf + enuc,
             ediff_rel, rms_error, time_elapsed.count());

    } while (((ediff_rel > conv) || (rms_error > conv)) && (iter < maxiter));

    auto Mu = compute_1body_ints<libint2::OneBodyEngine::emultipole2>(obs);
    std::array<double,3> mu;
    std::array<double,6> qu;
    for(int xyz=0; xyz!=3; ++xyz)
      mu[xyz] = -2 * D.cwiseProduct(Mu[xyz+1]).sum(); // 2 = alpha + beta, -1 = electron charge
    for(int k=0; k!=6; ++k)
      qu[k] = -2 * D.cwiseProduct(Mu[k+4]).sum(); // 2 = alpha + beta, -1 = electron charge
    std::cout << "** edipole = "; std::copy(mu.begin(), mu.end(), std::ostream_iterator<double>(std::cout, " ")); std::cout << std::endl;
    std::cout << "** equadrupole = ";std::copy(qu.begin(), qu.end(), std::ostream_iterator<double>(std::cout, " ")); std::cout << std::endl;

#if LIBINT2_DERIV_ONEBODY_ORDER
    // compute forces
    {
      Matrix F1 = Matrix::Zero(atoms.size(), 3);
      Matrix F_Pulay = Matrix::Zero(atoms.size(), 3);
      //////////
      // one-body contributions to the forces
      //////////
      auto T1 = compute_1body_deriv_ints<libint2::OneBodyEngine::kinetic>(1, obs, atoms);
      auto V1 = compute_1body_deriv_ints<libint2::OneBodyEngine::nuclear>(1, obs, atoms);
      for(auto atom=0, i=0; atom!=atoms.size(); ++atom) {
        for(auto xyz=0; xyz!=3; ++xyz, ++i) {
          auto force = 2 * (T1[i]+V1[i]).cwiseProduct(D).sum();
          F1(atom, xyz) += force;
//        std::cout << "one-body force=" << force << std::endl;
//        std::cout << "derivative nuclear ints:\n" << V1[i] << std::endl;
        }
      }

      //////////
      // Pulay force
      //////////
      // orbital energy density
      DiagonalMatrix evals_occ(evals.topRows(ndocc));
      Matrix W = C_occ * evals_occ * C_occ.transpose();
      auto S1 = compute_1body_deriv_ints<libint2::OneBodyEngine::overlap>(1, obs, atoms);
      for(auto atom=0, i=0; atom!=atoms.size(); ++atom) {
        for(auto xyz=0; xyz!=3; ++xyz, ++i) {
          auto force = 2 * S1[i].cwiseProduct(W).sum();
          F_Pulay(atom, xyz) -= force;
//        std::cout << "Pulay force=" << force << std::endl;
//        std::cout << "derivative overlap ints:\n" << S1[i] << std::endl;
        }
      }

      std::cout << "** 1-body forces = ";
      for(int atom=0; atom!=atoms.size(); ++atom)
        for(int xyz=0; xyz!=3; ++xyz)
          std::cout << F1(atom,xyz) << " ";
      std::cout << std::endl;
      std::cout << "** Pulay forces = ";
      for(int atom=0; atom!=atoms.size(); ++atom)
        for(int xyz=0; xyz!=3; ++xyz)
          std::cout << F_Pulay(atom,xyz) << " ";
      std::cout << std::endl;
    }
#endif

    printf("** Hartree-Fock energy = %20.12f\n", ehf + enuc);

    auto ktime = 0.;
    for(auto t : k_times){
        ktime += t;
    }
    std::cout << "Average K time = " << ktime/k_times.size() << std::endl;
    auto sum = 0.;
    for(auto t : j_ints){
        sum += double(t)/double(j_ints.size());
    }
    std::cout << "Average J Integrals = " << sum << std::endl;
    sum = 0.;
    for(auto t : k_ints){
        sum += double(t)/double(k_ints.size());
    }
    std::cout << "Average K Integrals = " << sum << std::endl;
    libint2::cleanup(); // done with libint

    std::cout << "Job Summary\n";
    std::cout << "nbasis, "
        << "basis, "
        << "nthreads, "
        << "energy, "
        << "k time, "
        << "precision J, "
        << "precision K, "
        << "build type"
        << std::endl;

    std::cout << obs.nbf() << ", " 
        << basisname << ", "
        << libint2::nthreads << ", "
        << ehf + enuc << ", "
        << ktime/k_times.size() << ", "
        << j_precision << ", "
        << k_precision << ", ";
    if(use_linK){
        std::cout << "link" << std::endl;
    } else {
        std::cout << "n2" << std::endl;
    }


  } // end of try block; if any exceptions occurred, report them and exit cleanly

  catch (const char* ex) {
    cerr << "caught exception: " << ex << endl;
    return 1;
  }
  catch (std::string& ex) {
    cerr << "caught exception: " << ex << endl;
    return 1;
  }
  catch (std::exception& ex) {
    cerr << ex.what() << endl;
    return 1;
  }
  catch (...) {
    cerr << "caught unknown exception\n";
    return 1;
  }

  return 0;
}

std::vector<Atom> read_geometry(const std::string& filename) {

  std::cout << "Will read geometry from " << filename << std::endl;
  std::ifstream is(filename);
  assert(is.good());

  // to prepare for MPI parallelization, we will read the entire file into a string that can be
  // broadcast to everyone, then converted to an std::istringstream object that can be used just like std::ifstream
  std::ostringstream oss;
  oss << is.rdbuf();
  // use ss.str() to get the entire contents of the file as an std::string
  // broadcast
  // then make an std::istringstream in each process
  std::istringstream iss(oss.str());

  // check the extension: if .xyz, assume the standard XYZ format, otherwise throw an exception
  if ( filename.rfind(".xyz") != std::string::npos)
    return libint2::read_dotxyz(iss);
  else
    throw "only .xyz files are accepted";
}

// computes Superposition-Of-Atomic-Densities guess for the molecular density matrix
// in minimal basis; occupies subshells by smearing electrons evenly over the orbitals
Matrix compute_soad(const std::vector<Atom>& atoms) {

  // compute number of atomic orbitals
  size_t nao = 0;
  for(const auto& atom: atoms) {
    const auto Z = atom.atomic_number;
    if (Z == 1 || Z == 2) // H, He
      nao += 1;
    else if (Z <= 10) // Li - Ne
      nao += 5;
    else
      throw "SOAD with Z > 10 is not yet supported";
  }

  // compute the minimal basis density
  Matrix D = Matrix::Zero(nao, nao);
  size_t ao_offset = 0; // first AO of this atom
  for(const auto& atom: atoms) {
    const auto Z = atom.atomic_number;
    if (Z == 1 || Z == 2) { // H, He
      D(ao_offset, ao_offset) = Z; // all electrons go to the 1s
      ao_offset += 1;
    }
    else if (Z <= 10) {
      D(ao_offset, ao_offset) = 2; // 2 electrons go to the 1s
      D(ao_offset+1, ao_offset+1) = (Z == 3) ? 1 : 2; // Li? only 1 electron in 2s, else 2 electrons
      // smear the remaining electrons in 2p orbitals
      const double num_electrons_per_2p = (Z > 4) ? (double)(Z - 4)/3 : 0;
      for(auto xyz=0; xyz!=3; ++xyz)
        D(ao_offset+2+xyz, ao_offset+2+xyz) = num_electrons_per_2p;
      ao_offset += 5;
    }
  }

  return D * 0.5; // we use densities normalized to # of electrons/2
}

Matrix compute_shellblock_norm(const BasisSet& obs,
                               const Matrix& A) {
  const auto nsh = obs.size();
  Matrix Ash(nsh, nsh);

  auto shell2bf = obs.shell2bf();
  for(size_t s1=0; s1!=nsh; ++s1) {
    const auto& s1_first = shell2bf[s1];
    const auto& s1_size = obs[s1].size();
    for(size_t s2=0; s2!=nsh; ++s2) {
      const auto& s2_first = shell2bf[s2];
      const auto& s2_size = obs[s2].size();

      Ash(s1, s2) = A.block(s1_first, s2_first, s1_size, s2_size).lpNorm<Eigen::Infinity>();
    }
  }

  return Ash;
}

template <libint2::OneBodyEngine::operator_type obtype>
std::array<Matrix, libint2::OneBodyEngine::operator_traits<obtype>::nopers>
compute_1body_ints(const BasisSet& obs,
                   const std::vector<Atom>& atoms)
{
  const auto n = obs.nbf();
  const auto nshells = obs.size();
#ifdef _OPENMP
  const auto nthreads = omp_get_max_threads();
#else
  const auto nthreads = 1;
#endif
  typedef std::array<Matrix, libint2::OneBodyEngine::operator_traits<obtype>::nopers> result_type;
  const unsigned int nopers = libint2::OneBodyEngine::operator_traits<obtype>::nopers;
  result_type result; for(auto& r: result) r = Matrix::Zero(n,n);

  // construct the 1-body integrals engine
  std::vector<libint2::OneBodyEngine> engines(nthreads);
  engines[0] = libint2::OneBodyEngine(obtype, obs.max_nprim(), obs.max_l(), 0);
  // nuclear attraction ints engine needs to know where the charges sit ...
  // the nuclei are charges in this case; in QM/MM there will also be classical charges
  if (obtype == libint2::OneBodyEngine::nuclear) {
    std::vector<std::pair<double,std::array<double,3>>> q;
    for(const auto& atom : atoms) {
      q.push_back( {static_cast<double>(atom.atomic_number), {{atom.x, atom.y, atom.z}}} );
    }
    engines[0].set_params(q);
  }
  for(size_t i=1; i!=nthreads; ++i) {
    engines[i] = engines[0];
  }

  auto shell2bf = obs.shell2bf();

#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
#ifdef _OPENMP
    auto thread_id = omp_get_thread_num();
#else
    auto thread_id = 0;
#endif

    // loop over unique shell pairs, {s1,s2} such that s1 >= s2
    // this is due to the permutational symmetry of the real integrals over Hermitian operators: (1|2) = (2|1)
    for(auto s1=0l, s12=0l; s1!=nshells; ++s1) {

      auto bf1 = shell2bf[s1]; // first basis function in this shell
      auto n1 = obs[s1].size();

      for(auto s2=0; s2<=s1; ++s2) {

        if (s12 % nthreads != thread_id)
          continue;

        auto bf2 = shell2bf[s2];
        auto n2 = obs[s2].size();

        auto n12 = n1 * n2;

        // compute shell pair; return is the pointer to the buffer
        const auto* buf = engines[thread_id].compute(obs[s1], obs[s2]);

        for(unsigned int op=0; op!=nopers; ++op, buf+=n12) {
          // "map" buffer to a const Eigen Matrix, and copy it to the corresponding blocks of the result
          Eigen::Map<const Matrix> buf_mat(buf, n1, n2);
          result[op].block(bf1, bf2, n1, n2) = buf_mat;
          if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1} block, note the transpose!
            result[op].block(bf2, bf1, n2, n1) = buf_mat.transpose();
        }

      }
    }
  } // omp parallel

  return result;
}

#if LIBINT2_DERIV_ONEBODY_ORDER
template <libint2::OneBodyEngine::operator_type obtype>
std::vector<Matrix>
compute_1body_deriv_ints(unsigned deriv_order,
                         const BasisSet& obs,
                         const std::vector<Atom>& atoms)
{
  const auto n = obs.nbf();
  const auto nshells = obs.size();
#ifdef _OPENMP
  const auto nthreads = omp_get_max_threads();
#else
  const auto nthreads = 1;
#endif
  constexpr auto nopers = libint2::OneBodyEngine::operator_traits<obtype>::nopers;
  const auto nresults = nopers * libint2::num_geometrical_derivatives(atoms.size(),deriv_order);
  typedef std::vector<Matrix> result_type;
  result_type result(nresults); for(auto& r: result) r = Matrix::Zero(n,n);

  // construct the 1-body integrals engine
  std::vector<libint2::OneBodyEngine> engines(nthreads);
  engines[0] = libint2::OneBodyEngine(obtype, obs.max_nprim(), obs.max_l(), deriv_order);
  // nuclear attraction ints engine needs to know where the charges sit ...
  // the nuclei are charges in this case; in QM/MM there will also be classical charges
  if (obtype == libint2::OneBodyEngine::nuclear) {
    std::vector<std::pair<double,std::array<double,3>>> q;
    for(const auto& atom : atoms) {
      q.push_back( {static_cast<double>(atom.atomic_number), {{atom.x, atom.y, atom.z}}} );
    }
    engines[0].set_params(q);
  }
  for(size_t i=1; i!=nthreads; ++i) {
    engines[i] = engines[0];
  }

  auto shell2bf = obs.shell2bf();
  auto shell2atom = obs.shell2atom(atoms);

#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
#ifdef _OPENMP
    auto thread_id = omp_get_thread_num();
#else
    auto thread_id = 0;
#endif

    // loop over unique shell pairs, {s1,s2} such that s1 >= s2
    // this is due to the permutational symmetry of the real integrals over Hermitian operators: (1|2) = (2|1)
    for(auto s1=0l, s12=0l; s1!=nshells; ++s1) {

      auto bf1 = shell2bf[s1]; // first basis function in this shell
      auto n1 = obs[s1].size();
      auto atom1 = shell2atom[s1];
      assert(atom1 != -1);

      for(auto s2=0; s2<=s1; ++s2) {

        if (s12 % nthreads != thread_id)
          continue;

        auto bf2 = shell2bf[s2];
        auto n2 = obs[s2].size();
        auto atom2 = shell2atom[s2];

        auto n12 = n1 * n2;

        // compute shell pair; return is the pointer to the buffer
        const auto* buf = engines[thread_id].compute(obs[s1], obs[s2]);

        assert(deriv_order == 1); // the loop structure below needs to be generalized for higher-order derivatives
        // 1. process derivatives with respect to the Gaussian origins first ...
        for(unsigned int d=0; d!=6; ++d) { // 2 centers x 3 axes = 6 cartesian geometric derivatives
          auto atom = d < 3 ? atom1 : atom2;
          auto op_start = (3*atom+d%3) * nopers;
          auto op_fence = op_start + nopers;
          for(unsigned int op=op_start; op!=op_fence; ++op, buf+=n12) {
            // "map" buffer to a const Eigen Matrix, and copy it to the corresponding blocks of the result
            Eigen::Map<const Matrix> buf_mat(buf, n1, n2);
//            std::cout << "s1=" << s1 << " s2=" << s2 << " x=" << (3*atom+d%3) << " op=" << op-op_start << ":\n";
//            std::cout << buf_mat << std::endl;
            result[op].block(bf1, bf2, n1, n2) += buf_mat;
            if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1} block, note the transpose!
              result[op].block(bf2, bf1, n2, n1) += buf_mat.transpose();
          }
        }
        // 2. process derivatives of nuclear Coulomb operator, if needed
        if (obtype == libint2::OneBodyEngine::nuclear) {
          for(unsigned int atom=0; atom!=atoms.size(); ++atom) {
            for(unsigned int xyz=0; xyz!=3; ++xyz) {
              auto op_start = (3*atom+xyz) * nopers;
              auto op_fence = op_start + nopers;
              for(unsigned int op=op_start; op!=op_fence; ++op, buf+=n12) {
                Eigen::Map<const Matrix> buf_mat(buf, n1, n2);
                result[op].block(bf1, bf2, n1, n2) += buf_mat;
                if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1} block, note the transpose!
                  result[op].block(bf2, bf1, n2, n1) += buf_mat.transpose();
              }
            }
          }
        }

      }
    }
  } // omp parallel

  return result;
}
#endif

Matrix compute_schwartz_ints(const BasisSet& bs1,
                             const BasisSet& _bs2,
                             bool use_2norm) {
  const BasisSet& bs2 = (_bs2.empty() ? bs1 : _bs2);
  const auto nsh1 = bs1.size();
  const auto nsh2 = bs2.size();
  const auto bs1_equiv_bs2 = (&bs1 == &bs2);

  Matrix K = Matrix::Zero(nsh1,nsh2);
#ifdef _OPENMP
  const auto nthreads = omp_get_max_threads();
#else
  const auto nthreads = 1;
#endif

  // construct the 2-electron repulsion integrals engine
  typedef libint2::TwoBodyEngine<libint2::Coulomb> coulomb_engine_type;
  std::vector<coulomb_engine_type> engines(nthreads);
  engines[0] = coulomb_engine_type(bs1.max_nprim(), bs2.max_l(), 0);
  engines[0].set_precision(0.); // !!! very important: cannot screen primitives in Schwartz computation !!!
  for(size_t i=1; i!=nthreads; ++i) {
    engines[i] = engines[0];
  }

  std::cout << "computing Schwartz bound prerequisites ... ";

  libint2::Timers<1> timer;
  timer.set_now_overhead(25);
  timer.start(0);

#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
#ifdef _OPENMP
    auto thread_id = omp_get_thread_num();
#else
    auto thread_id = 0;
#endif

    // loop over permutationally-unique set of shells
    for(auto s1=0l, s12=0l; s1!=nsh1; ++s1) {

      auto n1 = bs1[s1].size();// number of basis functions in this shell

      auto s2_max = bs1_equiv_bs2 ? s1 : nsh2-1;
      for(auto s2=0; s2<=s2_max; ++s2, ++s12) {

        if (s12 % nthreads != thread_id)
          continue;

        auto n2 = bs2[s2].size();
        auto n12 = n1*n2;

        const auto* buf = engines[thread_id].compute(bs1[s1], bs2[s2], bs1[s1], bs2[s2]);

        // the diagonal elements are the Schwartz ints ... use Map.diagonal()
        Eigen::Map<const Matrix> buf_mat(buf, n12, n12);
        auto norm2 = use_2norm ? buf_mat.diagonal().norm() 
            : buf_mat.diagonal().lpNorm<Eigen::Infinity>();
        K(s1,s2) = std::sqrt(norm2);
        if (bs1_equiv_bs2) K(s2,s1) = K(s1,s2);

      }
    }
  }

  timer.stop(0);
  std::cout << "done (" << timer.read(0) << " s)"<< std::endl;

  return K;
}

Matrix compute_do_ints(const BasisSet& bs1,
                       const BasisSet& _bs2,
                       bool use_2norm) {
  const BasisSet& bs2 = (_bs2.empty() ? bs1 : _bs2);
  const auto nsh1 = bs1.size();
  const auto nsh2 = bs2.size();
  const auto bs1_equiv_bs2 = (&bs1 == &bs2);

  Matrix K = Matrix::Zero(nsh1,nsh2);
#ifdef _OPENMP
  const auto nthreads = omp_get_max_threads();
#else
  const auto nthreads = 1;
#endif

  // construct the 2-electron repulsion integrals engine
  typedef libint2::TwoBodyEngine<libint2::Delta> coulomb_engine_type;
  std::vector<coulomb_engine_type> engines(nthreads);
  engines[0] = coulomb_engine_type(bs1.max_nprim(), bs2.max_l(), 0);
  engines[0].set_precision(0.); // !!! very important: cannot screen primitives in Schwartz computation !!!
  for(size_t i=1; i!=nthreads; ++i) {
    engines[i] = engines[0];
  }

  std::cout << "computing DOIs ... ";

  libint2::Timers<1> timer;
  timer.set_now_overhead(25);
  timer.start(0);

#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
#ifdef _OPENMP
    auto thread_id = omp_get_thread_num();
#else
    auto thread_id = 0;
#endif

    // loop over permutationally-unique set of shells
    for(auto s1=0l, s12=0l; s1!=nsh1; ++s1) {

      auto n1 = bs1[s1].size();// number of basis functions in this shell

      auto s2_max = bs1_equiv_bs2 ? s1 : nsh2-1;
      for(auto s2=0; s2<=s2_max; ++s2, ++s12) {

        if (s12 % nthreads != thread_id)
          continue;

        auto n2 = bs2[s2].size();
        auto n12 = n1*n2;

        const auto* buf = engines[thread_id].compute(bs1[s1], bs2[s2], bs1[s1], bs2[s2]);

        // the diagonal elements are the Schwartz ints ... use Map.diagonal()
        Eigen::Map<const Matrix> buf_mat(buf, n12, n12);
        auto norm2 = use_2norm ? buf_mat.diagonal().norm() : buf_mat.diagonal().lpNorm<Eigen::Infinity>();
        K(s1,s2) = std::sqrt(norm2);
        if (bs1_equiv_bs2) K(s2,s1) = K(s1,s2);

      }
    }
  }

  timer.stop(0);
  std::cout << "done (" << timer.read(0) << " s)"<< std::endl;

  return K;
}

shellpair_list_t
compute_shellpair_list(const BasisSet& bs1,
                       const BasisSet& _bs2,
                       const double threshold) {
  const BasisSet& bs2 = (_bs2.empty() ? bs1 : _bs2);
  const auto nsh1 = bs1.size();
  const auto nsh2 = bs2.size();
  const auto bs1_equiv_bs2 = (&bs1 == &bs2);

#ifdef _OPENMP
  const auto nthreads = omp_get_max_threads();
#else
  const auto nthreads = 1;
#endif

  // construct the 2-electron repulsion integrals engine
  using libint2::OneBodyEngine;
  std::vector<OneBodyEngine> engines; engines.reserve(nthreads);
  engines.emplace_back(OneBodyEngine::overlap,
                       std::max(bs1.max_nprim(),bs2.max_nprim()),
                       std::max(bs1.max_l(),bs2.max_l()),
                       0);
  for(size_t i=1; i!=nthreads; ++i) {
    engines.push_back(engines[0]);
  }

  std::cout << "computing non-negligible shell-pair list ... ";

  libint2::Timers<1> timer;
  timer.set_now_overhead(25);
  timer.start(0);

  shellpair_list_t result;

  std::mutex mx;

  auto compute = [&] (int thread_id) {

    auto& engine = engines[thread_id];

    // loop over permutationally-unique set of shells
    for(auto s1=0l, s12=0l; s1!=nsh1; ++s1) {

      mx.lock();
      if (result.find(s1) == result.end())
        result.insert(std::make_pair(s1,std::vector<size_t>()));
      mx.unlock();

      auto n1 = bs1[s1].size();// number of basis functions in this shell

      auto s2_max = bs1_equiv_bs2 ? s1 : nsh2-1;
      for(auto s2=0; s2<=s2_max; ++s2, ++s12) {

        if (s12 % nthreads != thread_id)
          continue;

        auto on_same_center = (bs1[s1].O == bs2[s2].O);
        bool significant = on_same_center;
        if (not on_same_center) {
          auto n2 = bs2[s2].size();
          const auto* buf = engines[thread_id].compute(bs1[s1], bs2[s2]);
          Eigen::Map<const Matrix> buf_mat(buf, n1, n2);
          auto norm = buf_mat.norm();
          significant = (norm >= threshold);
        }

        if (significant) {
          mx.lock();
          result[s1].emplace_back(s2);
          mx.unlock();
        }
      }
    }
  }; // end of compute

  libint2::parallel_do(compute);

  // resort shell list
  auto sort = [&] (int thread_id) {
    for(auto s1=0l; s1!=nsh1; ++s1) {
      if (s1%nthreads == thread_id) {
        auto& list = result[s1];
        std::sort(list.begin(), list.end());
      }
    }
  }; // end of sort

  libint2::parallel_do(sort);

  timer.stop(0);
  std::cout << "done (" << timer.read(0) << " s)"<< std::endl;

  return result;
}

Matrix compute_2body_2index_ints(const BasisSet& bs)
{
  const auto n = bs.nbf();
  const auto nshells = bs.size();
#ifdef _OPENMP
  const auto nthreads = omp_get_max_threads();
#else
  const auto nthreads = 1;
#endif
  Matrix result = Matrix::Zero(n,n);

  // build engines for each thread
  typedef libint2::TwoBodyEngine<libint2::Coulomb> coulomb_engine_type;
  std::vector<coulomb_engine_type> engines(nthreads);
  engines[0] = coulomb_engine_type(bs.max_nprim(), bs.max_l(), 0);
  for(size_t i=1; i!=nthreads; ++i) {
    engines[i] = engines[0];
  }

  auto shell2bf = bs.shell2bf();
  auto unitshell = Shell::unit();

#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
#ifdef _OPENMP
    auto thread_id = omp_get_thread_num();
#else
    auto thread_id = 0;
#endif

    // loop over unique shell pairs, {s1,s2} such that s1 >= s2
    // this is due to the permutational symmetry of the real integrals over Hermitian operators: (1|2) = (2|1)
    for(auto s1=0l, s12=0l; s1!=nshells; ++s1) {

      auto bf1 = shell2bf[s1]; // first basis function in this shell
      auto n1 = bs[s1].size();

      for(auto s2=0; s2<=s1; ++s2) {

        if (s12 % nthreads != thread_id)
          continue;

        auto bf2 = shell2bf[s2];
        auto n2 = bs[s2].size();

        // compute shell pair; return is the pointer to the buffer
        const auto* buf = engines[thread_id].compute(bs[s1], unitshell, bs[s2], unitshell);

        // "map" buffer to a const Eigen Matrix, and copy it to the corresponding blocks of the result
        Eigen::Map<const Matrix> buf_mat(buf, n1, n2);
        result.block(bf1, bf2, n1, n2) = buf_mat;
        if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1} block, note the transpose!
        result.block(bf2, bf1, n2, n1) = buf_mat.transpose();

      }
    }
  } // omp parallel

  return result;
}

Matrix compute_2body_fock_fake(const BasisSet& obs,
                          const Matrix& D,
                          double precision,
                          const Matrix& Schwartz,
                          bool use_linK) {

  const auto n = obs.nbf();
  const auto nshells = obs.size();
  using libint2::nthreads;
  std::vector<Matrix> G(nthreads, Matrix::Zero(n,n));

  const auto do_schwartz_screen = Schwartz.cols() != 0 && Schwartz.rows() != 0;
  Matrix D_shblk_norm; // matrix of norms of shell blocks
  if (do_schwartz_screen) {
    D_shblk_norm = compute_shellblock_norm(obs, D);
  }

  // construct the 2-electron repulsion integrals engine
  typedef libint2::TwoBodyEngine<libint2::Coulomb> coulomb_engine_type;
  std::vector<coulomb_engine_type> engines(nthreads);
  engines[0] = coulomb_engine_type(obs.max_nprim(), obs.max_l(), 0);
  engines[0].set_precision(std::min(precision,std::numeric_limits<double>::epsilon())); // shellset-dependent precision control will likely break positive definiteness
                                       // stick with this simple recipe
  std::cout << "compute_2body_fock:precision = " << precision << std::endl;
  std::cout << "TwoBodyEngine::precision = " << engines[0].precision() << std::endl;
  for(size_t i=1; i!=nthreads; ++i) {
    engines[i] = engines[0];
  }
  std::atomic<std::size_t> num_j_computed{0};
  std::atomic<std::size_t> num_k_computed{0};
  std::atomic<std::size_t> num_k_loop_checked{0};
  std::atomic<std::size_t> num_k_loop_computed{0};
  std::atomic<std::size_t> num_k_loop_inserted{0};

#if defined(REPORT_INTEGRAL_TIMINGS)
  std::vector<libint2::Timers<1>> timers(nthreads);
#endif

  auto shell2bf = obs.shell2bf();

  namespace tim = std::chrono;
  using dur = tim::duration<double>;

  // From 1998 LinK paper Ochsenfeld, White and Head-Gordon
  // 4 center int is (\mu \lambda | \nu \sigma) 
  

  Matrix const &Q = Schwartz;
  Matrix const &Ds = D_shblk_norm;

  // Compute max vals in rows of Q but don't time for now
  std::vector<double> max_Q(nshells);
  std::vector<double> max_D(nshells);
  for(auto i = 0; i < Q.rows(); ++i){
      max_Q[i] = Q.lpNorm<Eigen::Infinity>();
      max_D[i] = Ds.lpNorm<Eigen::Infinity>();
  }

  auto prescreen0 = tim::high_resolution_clock::now();

#if 1 // LinK style loops
  std::vector<std::vector<std::pair<int,double>>> sig_nu_in_mus(nshells);
  // Loop over all significant \mu and all significant \nu 
  for(auto s1 = 0; s1 != nshells; ++s1){
      for(auto s3=0; s3 != nshells; ++s3) {
          auto d_nu_bound = Ds(s1, s3) * max_Q[s3];
          if(max_Q[s1] * d_nu_bound >= precision){
              sig_nu_in_mus[s1].push_back(std::make_pair(s3,d_nu_bound));
          }
      }

      // Sort \nu by size of D(\mu,\nu) * max_Q[\nu]
      auto start = sig_nu_in_mus[s1].begin();
      auto end = sig_nu_in_mus[s1].end();
      std::sort(start, end, [](std::pair<int, double> const& a,
                               std::pair<int, double> const b) {
          // Sort greatest to least
          return b.second < a.second;
      });
  }

  using sorted_shell_pair_list = 
      std::unordered_map<std::size_t, std::vector<std::pair<int, double>>>;

  sorted_shell_pair_list a_sorted_for;
  for (auto i = 0; i < nshells; ++i) {
      std::vector<std::pair<int, double>> sign_sig;
      for (auto j = 0; j < nshells; ++j) {
          sign_sig.push_back(std::make_pair(j, Q(i, j)));
      }
      std::sort(
          sign_sig.begin(), sign_sig.end(),
          [](std::pair<int, double> const& a, std::pair<int, double> const b) {
              // Sort greatest to least
              return b.second < a.second;
          });
      a_sorted_for[i] = std::move(sign_sig);
  }

#endif

#if 0 // Drew's significant s3 for S1 builder
  // Vector to hold significant \nu for every \mu
  std::vector<std::vector<int>> sig_nu_in_mus(nshells);
  for (auto s1 = 0; s1 != nshells; ++s1) {
      for (auto s2 : obs_shellpair_list[s1]) {
          const auto Q12 = Q(s1, s2);

          auto maxD = 0.0;
          for (auto s3 = 0; s3 <= s1; ++s3) {
              // Don't add s3 twice
              auto iter = std::find(sig_nu_in_mus[s1].begin(),
                      sig_nu_in_mus[s1].end(), s3);
              if (iter == sig_nu_in_mus[s1].end()) {
                  // maxD is actually the max possible val for Ds(s1,s3), Ds(s1,s4),
                  // Ds(s2,s3), Ds(s2,s4)
                  maxD = std::max({maxD, Ds(s1, s3), Ds(s2, s3)});
                  const auto maxS3D = maxD * max_Q[s3];
                  auto est = Q12 * maxS3D;

                  if (est >= precision) {
                      sig_nu_in_mus[s1].push_back(s3);
                  }
              }
          }
      }
  }

  using sorted_shell_pair_list = 
      std::unordered_map<std::size_t, std::vector<std::pair<int, double>>>;

  sorted_shell_pair_list nu_sig_sorted;
  for (auto i = 0; i < nshells; ++i) {
      std::vector<std::pair<int, double>> sign_sig;
      for (auto const& j : obs_shellpair_list[i]) {
          sign_sig.push_back(std::make_pair(j, Q(i, j)));
      }
      std::sort(
          sign_sig.begin(), sign_sig.end(),
          [](std::pair<int, double> const& a, std::pair<int, double> const b) {
              // Sort greatest to least
              return b.second < a.second;
          });
      nu_sig_sorted[i] = std::move(sign_sig);
  }
#endif

  auto prescreen1 = tim::high_resolution_clock::now();
  dur pstime = prescreen1 - prescreen0;

  auto lambdaJ = [&] (int thread_id) {

    auto& engine = engines[thread_id];
    auto& g = G[thread_id];

#if defined(REPORT_INTEGRAL_TIMINGS)
    auto& timer = timers[thread_id];
    timer.clear();
    timer.set_now_overhead(25);
#endif

    // loop over permutationally-unique set of shells
    for(auto s1=0l, s12 = 0l; s1!=nshells; ++s1) {
      auto bf1_first = shell2bf[s1]; // first basis function in this shell
      auto n1 = obs[s1].size();// number of basis functions in this shell

      for(const auto& s2: obs_shellpair_list[s1]) {
        if(s12++ % nthreads != thread_id)
            continue;

        auto bf2_first = shell2bf[s2];
        auto n2 = obs[s2].size();

        const auto Dnorm12 = do_schwartz_screen ? D_shblk_norm(s1,s2) : 0.;
        const auto Q12 = do_schwartz_screen ? Q(s1,s2) : 0.;

        for(auto s3=0; s3<=s1; ++s3) {
          auto bf3_first = shell2bf[s3];
          auto n3 = obs[s3].size();
          
          const auto s4_max = (s1 == s3) ? s2 : s3;
          for(const auto& s4: obs_shellpair_list[s3]) {
            if (s4 > s4_max) break; // for each s3, s4 are stored in monotonically increasing order

            const auto Qest = Q12 * Q(s3,s4);
            const auto est12 = Dnorm12 * Qest;
            const auto est34 = Ds(s3,s4) * Qest;

            if (do_schwartz_screen && std::max(est12, est34) < precision)
              continue;

            auto bf4_first = shell2bf[s4];
            auto n4 = obs[s4].size();

            num_j_computed += n1*n2*n3*n4;
          }
        }
      }
    }

  }; // end of lambda

  // Basically a hash table where each bin is an s3 and the inner vectors are s4s
  using pair_set = std::vector<std::vector<int>>;
  
  // Vector to hold sig kets for each thread
  std::vector<pair_set> sig_kets_vec(nthreads);
  for(auto i = 0; i < nthreads; ++i){
      sig_kets_vec[i] = pair_set(nshells);
  }

  // Reserve the maximum amount of space that each s3 could need interms of s4s
  for(auto s1 = 0; s1 < nshells; ++s1){
      auto size = 0;
      for(auto s2 : obs_shellpair_list[s1]){
         ++size;
      }

      for(auto i = 0; i < nthreads; ++i){
         sig_kets_vec[i][s1].reserve(size);
      }
  }

  // Vectors that tell us if there is a significant ket for s3
  std::vector<std::vector<int>> computed_s3s_vec(nthreads);
  for(auto i = 0; i < nthreads; ++i){
      computed_s3s_vec[i] = std::vector<int>(nshells, 0);
  }

  // Vector that actually holds the index of the significant s3s
  std::vector<std::vector<int>> sig_s3s_vec(nthreads);
  for(auto i = 0; i < nthreads; ++i){
      sig_s3s_vec[i].reserve(nshells);
  }

  // Matrices that hold the indices of significant s3 s4 pairs
  std::vector<MatrixI> comp_vec(nthreads);
  for(auto &comp : comp_vec){
      comp = MatrixI::Zero(nshells, nshells);
  }

  auto insert = [](std::vector<std::vector<int>>& v, std::vector<int>& sig_v,
                   MatrixI &comp,  std::vector<int> &computed_s3s, int s3, int s4) {
      auto& s3_ket = v[s3];
      if (!comp(s3, s4)) {
          s3_ket.push_back(s4);
          comp(s3, s4) = 1;
      }
      if (!computed_s3s[s3]) {
          sig_v.push_back(s3);
          computed_s3s[s3] = 1;
      }
  };

  auto clear = [](std::vector<std::vector<int>> &v, std::vector<int> &sig_v){
      for(auto i : sig_v){
        v[i].clear();
      }
      sig_v.clear();
  };

  auto lambdaK = [&](int thread_id) {

      auto& engine = engines[thread_id];
      auto& g = G[thread_id];
      auto& kets = sig_kets_vec[thread_id];
      auto& comp = comp_vec[thread_id];
      auto& computed_s3s = computed_s3s_vec[thread_id];
      auto& sig_s3s = sig_s3s_vec[thread_id];

#if defined(REPORT_INTEGRAL_TIMINGS)
      auto& timer = timers[thread_id];
      timer.clear();
      timer.set_now_overhead(25);
#endif
      if (use_linK) {
#if 1 // LinK style looping
          // significant \nu \sigma for \mu and \lambda

          for (auto s1 = 0l, s12 = 0l; s1 != nshells; ++s1) {
                // if(s1 % nthreads != thread_id) continue;
              auto bf1_first = shell2bf[s1];
              auto n1 = obs[s1].size();

              for (const auto& s2 : obs_shellpair_list[s1]) {
                if(s12++ % nthreads != thread_id)
                    continue;

                  auto bf2_first = shell2bf[s2];
                  auto n2 = obs[s2].size();

                  const auto Q12 = Q(s1, s2);

                  clear(kets, sig_s3s);

                  for (auto const& nu : sig_nu_in_mus[s1]) {
                      const auto s3 = nu.first;

                      if(s3 > s1) continue;
                      if(Q12 * nu.second < precision){
                          break; 
                      }

                      const auto D13 = Ds(s1, s3);

                      const auto s4_max = (s1 == s3) ? s2 : s3;
                      for (auto const& a_pair : a_sorted_for[s3]) {
                          auto s4 = a_pair.first;

                          if(s4 > s1) continue;

                          const auto Q34 = Q(s3,s4);
                          ++num_k_loop_checked;
                          if(Q12 * D13 * Q34 >= precision){
                              if(s4 <= s4_max){
                                  ++num_k_loop_inserted;
                                  insert(kets, sig_s3s, comp, computed_s3s, s3, s4);
                              }

                              const auto flip_max = (s1 == s4) ? s2 : s4;
                              if(s3 <= flip_max){
                                  ++num_k_loop_inserted;
                                  insert(kets, sig_s3s, comp, computed_s3s, s4, s3);
                              }
                          } else {
                              break;
                          }
                      }
                  }

                  if (s2 != s1) {
                      for (auto const& nu : sig_nu_in_mus[s2]) {
                          const auto s3 = nu.first;
                          if (s3 > s1) continue;
                          if (Q12 * nu.second < precision) {
                              break;
                          }

                          const auto D23 = Ds(s2, s3);

                          const auto s4_max = (s1 == s3) ? s2 : s3;
                          for (auto const& a_pair : a_sorted_for[s3]) {
                              auto s4 = a_pair.first;

                              if (s4 > s1) continue;

                              const auto Q34 = Q(s3, s4);
                              ++num_k_loop_checked;
                              if (Q12 * D23 * Q34 >= precision) {
                                  if (s4 <= s4_max) {
                                      ++num_k_loop_inserted;
                                      insert(kets, sig_s3s, comp, computed_s3s,
                                             s3, s4);
                                  }
                                  const auto flip_max = (s1 == s4) ? s2 : s4;
                                  if (s3 <= flip_max) {
                                      ++num_k_loop_inserted;
                                      insert(kets, sig_s3s, comp, computed_s3s,
                                             s4, s3);
                                  }
                              } else {
                                  break;
                              }
                          }
                      }
                  }

                  auto s12_deg = (s1 == s2) ? 1.0 : 2.0;

                  for(auto s3 : sig_s3s) {
                      computed_s3s[s3] = 0;
                      auto bf3_first = shell2bf[s3];
                      auto n3 = obs[s3].size();

                      for(auto s4 : kets[s3]){
                          ++num_k_loop_computed;
                          comp(s3,s4) = 0;

                      auto bf4_first = shell2bf[s4];
                      auto n4 = obs[s4].size();

                      num_k_computed += n1 * n2 * n3 * n4;

                      // compute the permutational degeneracy (i.e. # of
                      // equivalents)
                      // of the given shell set
                      auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
                      auto s12_34_deg =
                          (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;
                      auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

                  }
                  }
              }
          }
#endif
      } else {
          // loop over permutationally-unique set of shells
          for (auto s1 = 0l, s12 = 0l; s1 != nshells; ++s1) {
               //  if(s1 % nthreads != thread_id) continue;

              auto bf1_first = shell2bf[s1];  // first basis function in this shell
              auto n1 = obs[s1].size();  // number of basis functions in this shell

              for (const auto& s2 : obs_shellpair_list[s1]) {
                if(s12++ % nthreads != thread_id)
                    continue;
                  auto bf2_first = shell2bf[s2];
                  auto n2 = obs[s2].size();

                  const auto Dnorm12 = do_schwartz_screen ? D_shblk_norm(s1, s2) : 0.;
                  const auto Q12 = do_schwartz_screen ? Q(s1, s2) : 0.;

                  for (auto s3 = 0; s3 <= s1; ++s3) {
                      auto bf3_first = shell2bf[s3];
                      auto n3 = obs[s3].size();

                      const auto Dnorm123 = do_schwartz_screen
                                                ? std::max(D_shblk_norm(s1, s3),
                                                           D_shblk_norm(s2, s3))
                                                : 0.;

                      const auto s4_max = (s1 == s3) ? s2 : s3;
                      for (const auto& s4 : obs_shellpair_list[s3]) {
                          if (s4 > s4_max)
                              break;  // for each s3, s4 are stored in
                                      // monotonically increasing order
                                      
                          const auto Dnorm1234 =
                              do_schwartz_screen
                                  ? std::max(D_shblk_norm(s1, s4),
                                             std::max(D_shblk_norm(s2, s4),
                                                      Dnorm123))
                                  : 0.;


                          ++num_k_loop_checked;
                          if (do_schwartz_screen &&
                              Dnorm1234 * Q12 * Schwartz(s3, s4) < precision)
                              continue;

                          ++num_k_loop_computed;

                          auto bf4_first = shell2bf[s4];
                          auto n4 = obs[s4].size();

                          num_k_computed += n1 * n2 * n3 * n4;

                          // compute the permutational degeneracy (i.e. # of
                          // equivalents) of the given shell set
                          auto s12_deg = (s1 == s2) ? 1.0 : 2.0;
                          auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
                          auto s12_34_deg =
                              (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;
                          auto s1234_deg = s12_deg * s34_deg * s12_34_deg;
                      }
                  }
              }
          }
      }

  };  // end of lambda

  namespace tim = std::chrono;
  using dur = tim::duration<double>;

  auto j0 = tim::high_resolution_clock::now();
  libint2::parallel_do(lambdaJ);
  auto j1 = tim::high_resolution_clock::now();
  libint2::parallel_do(lambdaK);
  auto k1 = tim::high_resolution_clock::now();

  dur jtime = j1 - j0;
  dur ktime = k1 - j1;

  std::cout << "\tJ time = " << jtime.count() << std::endl;
  std::cout << "\tPrescreen time = " << pstime.count() << std::endl;
  std::cout << "\tK time = " << ktime.count() << std::endl;
  k_times.push_back(ktime.count());

  // accumulate contributions from all threads
  for(size_t i=1; i!=nthreads; ++i) {
    G[0] += G[i];
  }

#if defined(REPORT_INTEGRAL_TIMINGS)
  double time_for_ints = 0.0;
  for(auto& t: timers) {
    time_for_ints += t.read(0);
  }
  std::cout << "time for integrals = " << time_for_ints << std::endl;
  for(int t=0; t!=nthreads; ++t)
    engines[t].print_timers();
#endif

  Matrix GG = 0.5 * (G[0] + G[0].transpose());

  std::cout << "\t# of coloumb integrals = " << num_j_computed << std::endl;
  std::cout << "\t# of exchange integrals = " << num_k_computed << std::endl;
  std::cout << "\t# of exchange checked = " << num_k_loop_checked << std::endl;
  std::cout << "\t# of exchange inserted = " << num_k_loop_inserted << std::endl;
  std::cout << "\t# of exchange computed = " << num_k_loop_computed << std::endl;

  j_ints.push_back(num_j_computed);
  k_ints.push_back(num_k_computed);

  // symmetrize the result and return
  return GG;
}

Matrix compute_2body_fock(const BasisSet& obs,
                          const Matrix& D,
                          double j_precision,
                          double k_precision,
                          const Matrix& Schwartz,
                          bool use_linK) {

  const auto n = obs.nbf();
  const auto nshells = obs.size();
  using libint2::nthreads;
  std::vector<Matrix> G(nthreads, Matrix::Zero(n,n));

  const auto do_schwartz_screen = Schwartz.cols() != 0 && Schwartz.rows() != 0;
  Matrix D_shblk_norm; // matrix of norms of shell blocks
  if (do_schwartz_screen) {
    D_shblk_norm = compute_shellblock_norm(obs, D);
  }

  // construct the 2-electron repulsion integrals engine
  typedef libint2::TwoBodyEngine<libint2::Coulomb> coulomb_engine_type;
  std::vector<coulomb_engine_type> engines(nthreads);

  engines[0] = coulomb_engine_type(obs.max_nprim(), obs.max_l(), 0);
  engines[0].set_precision(std::numeric_limits<double>::epsilon()); // shellset-dependent precision control will likely break positive definiteness
                                       // stick with this simple recipe
                                       
  for(size_t i=1; i!=nthreads; ++i) {
    engines[i] = engines[0];
  }
  std::atomic<std::size_t> num_j_computed{0};
  std::atomic<std::size_t> num_k_computed{0};

#if defined(REPORT_INTEGRAL_TIMINGS)
  std::vector<libint2::Timers<1>> timers(nthreads);
#endif

  auto shell2bf = obs.shell2bf();

  namespace tim = std::chrono;
  using dur = tim::duration<double>;

  // From 1998 LinK paper Ochsenfeld, White and Head-Gordon
  // 4 center int is (\mu \lambda | \nu \sigma) 
  

  Matrix const &Q = Schwartz;
  Matrix const &Ds = D_shblk_norm;

  // Compute max vals in rows of Q but don't time for now
  std::vector<double> max_Q(nshells);
  std::vector<double> max_D(nshells);
  for(auto i = 0; i < Q.rows(); ++i){
      max_Q[i] = Q.lpNorm<Eigen::Infinity>();
      max_D[i] = Ds.lpNorm<Eigen::Infinity>();
  }

  auto prescreen0 = tim::high_resolution_clock::now();

#if 1 // LinK style loops
  std::vector<std::vector<std::pair<int,double>>> sig_nu_in_mus(nshells);
  // Loop over all significant \mu and all significant \nu 
  for(auto s1 = 0; s1 != nshells; ++s1){
      for(auto s3=0; s3 != nshells; ++s3) {
          auto d_nu_bound = Ds(s1, s3) * max_Q[s3];
          if(max_Q[s1] * d_nu_bound >= k_precision){
              sig_nu_in_mus[s1].push_back(std::make_pair(s3,d_nu_bound));
          }
      }

      // Sort \nu by size of D(\mu,\nu) * max_Q[\nu]
      auto start = sig_nu_in_mus[s1].begin();
      auto end = sig_nu_in_mus[s1].end();
      std::sort(start, end, [](std::pair<int, double> const& a,
                               std::pair<int, double> const b) {
          // Sort greatest to least
          return b.second < a.second;
      });
  }

  using sorted_shell_pair_list = 
      std::unordered_map<std::size_t, std::vector<std::pair<int, double>>>;

  sorted_shell_pair_list a_sorted_for;
  for (auto i = 0; i < nshells; ++i) {
      std::vector<std::pair<int, double>> sign_sig;
      for (auto j = 0; j < nshells; ++j) {
          sign_sig.push_back(std::make_pair(j, Q(i, j)));
      }
      std::sort(
          sign_sig.begin(), sign_sig.end(),
          [](std::pair<int, double> const& a, std::pair<int, double> const b) {
              // Sort greatest to least
              return b.second < a.second;
          });
      a_sorted_for[i] = std::move(sign_sig);
  }

#endif

  auto prescreen1 = tim::high_resolution_clock::now();
  dur pstime = prescreen1 - prescreen0;

  auto lambdaJ = [&] (int thread_id) {

    auto& engine = engines[thread_id];
    auto& g = G[thread_id];

#if defined(REPORT_INTEGRAL_TIMINGS)
    auto& timer = timers[thread_id];
    timer.clear();
    timer.set_now_overhead(25);
#endif

    // loop over permutationally-unique set of shells
    for(auto s1=0l, s12 = 0l; s1!=nshells; ++s1) {
      auto bf1_first = shell2bf[s1]; // first basis function in this shell
      auto n1 = obs[s1].size();// number of basis functions in this shell

// for(const auto& s2 : obs_shellpair_list[s1]) {
// Do J build "Exactly".
      for(auto s2 = 0l; s2 <= s1; ++s2){
        if(s12++ % nthreads != thread_id)
            continue;

        auto bf2_first = shell2bf[s2];
        auto n2 = obs[s2].size();

        const auto Dnorm12 = do_schwartz_screen ? D_shblk_norm(s1,s2) : 0.;
        const auto Q12 = do_schwartz_screen ? Q(s1,s2) : 0.;

        for(auto s3=0; s3<=s1; ++s3) {

          auto bf3_first = shell2bf[s3];
          auto n3 = obs[s3].size();
          
          const auto s4_max = (s1 == s3) ? s2 : s3;
          // Do J build exactly
          // for(const auto& s4: obs_shellpair_list[s3]) {
          for(auto s4 = 0; s4 <= s4_max; ++s4){
            // if (s4 > s4_max) break; // for each s3, s4 are stored in monotonically increasing order

            const auto Qest = Q12 * Q(s3,s4);
            const auto est12 = Dnorm12 * Qest;
            const auto est34 = Ds(s3,s4) * Qest;

            if (do_schwartz_screen && std::max(est12, est34) < j_precision)
              continue;

            auto bf4_first = shell2bf[s4];
            auto n4 = obs[s4].size();

            num_j_computed += n1*n2*n3*n4;

            // compute the permutational degeneracy (i.e. # of equivalents) of the given shell set
            auto s12_deg = (s1 == s2) ? 1.0 : 2.0;
            auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
            auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;
            auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

#if defined(REPORT_INTEGRAL_TIMINGS)
            timer.start(0);
#endif

            const auto* buf = engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);

#if defined(REPORT_INTEGRAL_TIMINGS)
            timer.stop(0);
#endif

            // 1) each shell set of integrals contributes up to 6 shell sets of the Fock matrix:
            //    F(a,b) += (ab|cd) * D(c,d)
            //    F(c,d) += (ab|cd) * D(a,b)
            // 2) each permutationally-unique integral (shell set) must be scaled by its degeneracy,
            //    i.e. the number of the integrals/sets equivalent to it
            // 3) the end result must be symmetrized
            for(auto f1=0, f1234=0; f1!=n1; ++f1) {
              const auto bf1 = f1 + bf1_first;
              for(auto f2=0; f2!=n2; ++f2) {
                const auto bf2 = f2 + bf2_first;
                for(auto f3=0; f3!=n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for(auto f4=0; f4!=n4; ++f4, ++f1234) {
                    const auto bf4 = f4 + bf4_first;

                    const auto value = buf[f1234];

                    const auto value_scal_by_deg = value * s1234_deg;

                    g(bf1,bf2) += D(bf3,bf4) * value_scal_by_deg;
                    g(bf3,bf4) += D(bf1,bf2) * value_scal_by_deg;
                  }
                }
              }
            }

          }
        }
      }
    }

  }; // end of lambda

  // Basically a hash table where each bin is an s3 and the inner vectors are s4s
  using pair_set = std::vector<std::vector<int>>;
  
  // Vector to hold sig kets for each thread
  std::vector<pair_set> sig_kets_vec(nthreads);
  for(auto i = 0; i < nthreads; ++i){
      sig_kets_vec[i] = pair_set(nshells);
  }

  // Reserve the maximum amount of space that each s3 could need interms of s4s
  for(auto s1 = 0; s1 < nshells; ++s1){
      auto size = 0;
      for(auto s2 : obs_shellpair_list[s1]){
         ++size;
      }

      for(auto i = 0; i < nthreads; ++i){
         sig_kets_vec[i][s1].reserve(size);
      }
  }

  // Vectors that tell us if there is a significant ket for s3
  std::vector<std::vector<int>> computed_s3s_vec(nthreads);
  for(auto i = 0; i < nthreads; ++i){
      computed_s3s_vec[i] = std::vector<int>(nshells, 0);
  }

  // Vector that actually holds the index of the significant s3s
  std::vector<std::vector<int>> sig_s3s_vec(nthreads);
  for(auto i = 0; i < nthreads; ++i){
      sig_s3s_vec[i].reserve(nshells);
  }

  // Matrices that hold the indices of significant s3 s4 pairs
  std::vector<MatrixI> comp_vec(nthreads);
  for(auto &comp : comp_vec){
      comp = MatrixI::Zero(nshells, nshells);
  }

  auto insert = [](std::vector<std::vector<int>>& v, std::vector<int>& sig_v,
                   MatrixI &comp,  std::vector<int> &computed_s3s, int s3, int s4) {
      auto& s3_ket = v[s3];
      if (!comp(s3, s4)) {
          s3_ket.push_back(s4);
          comp(s3, s4) = 1;
      }
      if (!computed_s3s[s3]) {
          sig_v.push_back(s3);
          computed_s3s[s3] = 1;
      }
  };

  auto clear = [](std::vector<std::vector<int>> &v, std::vector<int> &sig_v){
      for(auto i : sig_v){
        v[i].clear();
      }
      sig_v.clear();
  };

  auto lambdaK = [&](int thread_id) {

      auto& engine = engines[thread_id];
      auto& g = G[thread_id];
      auto& kets = sig_kets_vec[thread_id];
      auto& comp = comp_vec[thread_id];
      auto& computed_s3s = computed_s3s_vec[thread_id];
      auto& sig_s3s = sig_s3s_vec[thread_id];

#if defined(REPORT_INTEGRAL_TIMINGS)
      auto& timer = timers[thread_id];
      timer.clear();
      timer.set_now_overhead(25);
#endif
      if (use_linK) {
#if 1 // LinK style looping
          // significant \nu \sigma for \mu and \lambda

          for (auto s1 = 0l, s12 = 0l; s1 != nshells; ++s1) {
                // if(s1 % nthreads != thread_id) continue;
              auto bf1_first = shell2bf[s1];
              auto n1 = obs[s1].size();

              for (const auto& s2 : obs_shellpair_list[s1]) {
                if(s12++ % nthreads != thread_id)
                    continue;

                  auto bf2_first = shell2bf[s2];
                  auto n2 = obs[s2].size();

                  const auto Q12 = Q(s1, s2);

                  clear(kets, sig_s3s);

                  for (auto const& nu : sig_nu_in_mus[s1]) {
                      const auto s3 = nu.first;

                      if(s3 > s1) continue;
                      if(Q12 * nu.second < k_precision){
                          break; 
                      }

                      const auto D13 = Ds(s1, s3);

                      const auto s4_max = (s1 == s3) ? s2 : s3;
                      for (auto const& a_pair : a_sorted_for[s3]) {
                          auto s4 = a_pair.first;

                          if(s4 > s1) continue;

                          const auto Q34 = Q(s3,s4);
                          if(Q12 * D13 * Q34 >= k_precision){
                              if(s4 <= s4_max){
                                  insert(kets, sig_s3s, comp, computed_s3s, s3, s4);
                              }

                              const auto flip_max = (s1 == s4) ? s2 : s4;
                              if(s3 <= flip_max){
                                  insert(kets, sig_s3s, comp, computed_s3s, s4, s3);
                              }
                          } else {
                              break;
                          }
                      }
                  }

                  if (s2 != s1) {
                      for (auto const& nu : sig_nu_in_mus[s2]) {
                          const auto s3 = nu.first;
                          if (s3 > s1) continue;
                          if (Q12 * nu.second < k_precision) {
                              break;
                          }

                          const auto D23 = Ds(s2, s3);

                          const auto s4_max = (s1 == s3) ? s2 : s3;
                          for (auto const& a_pair : a_sorted_for[s3]) {
                              auto s4 = a_pair.first;

                              if (s4 > s1) continue;

                              const auto Q34 = Q(s3, s4);
                              if (Q12 * D23 * Q34 >= k_precision) {
                                  if (s4 <= s4_max) {
                                      insert(kets, sig_s3s, comp, computed_s3s,
                                             s3, s4);
                                  }
                                  const auto flip_max = (s1 == s4) ? s2 : s4;
                                  if (s3 <= flip_max) {
                                      insert(kets, sig_s3s, comp, computed_s3s,
                                             s4, s3);
                                  }
                              } else {
                                  break;
                              }
                          }
                      }
                  }

                  auto s12_deg = (s1 == s2) ? 1.0 : 2.0;

                  for(auto s3 : sig_s3s) {
                      computed_s3s[s3] = 0;
                      auto bf3_first = shell2bf[s3];
                      auto n3 = obs[s3].size();

                      for(auto s4 : kets[s3]){
                          comp(s3,s4) = 0;

                      auto bf4_first = shell2bf[s4];
                      auto n4 = obs[s4].size();

                      num_k_computed += n1 * n2 * n3 * n4;

                      // compute the permutational degeneracy (i.e. # of
                      // equivalents)
                      // of the given shell set
                      auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
                      auto s12_34_deg =
                          (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;
                      auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

#if defined(REPORT_INTEGRAL_TIMINGS)
                      timer.start(0);
#endif

                      const auto* buf =
                          engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);

#if defined(REPORT_INTEGRAL_TIMINGS)
                      timer.stop(0);
#endif

                      // 1) each shell set of integrals contributes up to 6
                      // shell
                      // sets
                      // of the Fock matrix:
                      //    F(b,d) -= 1/4 * (ab|cd) * D(a,c)
                      //    F(b,c) -= 1/4 * (ab|cd) * D(a,d)
                      //    F(a,c) -= 1/4 * (ab|cd) * D(b,d)
                      //    F(a,d) -= 1/4 * (ab|cd) * D(b,c)
                      // 2) each permutationally-unique integral (shell set)
                      // must be
                      // scaled by its degeneracy,
                      //    i.e. the number of the integrals/sets equivalent to
                      //    it
                      // 3) the end result must be symmetrized
                      for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                          const auto bf1 = f1 + bf1_first;
                          for (auto f2 = 0; f2 != n2; ++f2) {
                              const auto bf2 = f2 + bf2_first;
                              for (auto f3 = 0; f3 != n3; ++f3) {
                                  const auto bf3 = f3 + bf3_first;
                                  for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
                                      const auto bf4 = f4 + bf4_first;

                                      const auto value = buf[f1234];

                                      const auto value_scal_by_deg =
                                          value * s1234_deg;

                                      g(bf1, bf3) -= 0.25 * D(bf2, bf4) *
                                                     value_scal_by_deg;
                                      g(bf2, bf4) -= 0.25 * D(bf1, bf3) *
                                                     value_scal_by_deg;
                                      g(bf1, bf4) -= 0.25 * D(bf2, bf3) *
                                                     value_scal_by_deg;
                                      g(bf2, bf3) -= 0.25 * D(bf1, bf4) *
                                                     value_scal_by_deg;
                                  }
                              }
                          }
                      }
                  }
                  }
              }
          }
#endif
      } else {
          // loop over permutationally-unique set of shells
          for (auto s1 = 0l, s12 = 0l; s1 != nshells; ++s1) {
               //  if(s1 % nthreads != thread_id) continue;

              auto bf1_first = shell2bf[s1];  // first basis function in this shell
              auto n1 = obs[s1].size();  // number of basis functions in this shell

              for (const auto& s2 : obs_shellpair_list[s1]) {
                if(s12++ % nthreads != thread_id)
                    continue;
                  auto bf2_first = shell2bf[s2];
                  auto n2 = obs[s2].size();

                  const auto Dnorm12 = do_schwartz_screen ? D_shblk_norm(s1, s2) : 0.;
                  const auto Q12 = do_schwartz_screen ? Q(s1, s2) : 0.;

                  for (auto s3 = 0; s3 <= s1; ++s3) {
                      auto bf3_first = shell2bf[s3];
                      auto n3 = obs[s3].size();

                      const auto Dnorm123 = do_schwartz_screen
                                                ? std::max(D_shblk_norm(s1, s3),
                                                           D_shblk_norm(s2, s3))
                                                : 0.;

                      const auto s4_max = (s1 == s3) ? s2 : s3;
                      for (const auto& s4 : obs_shellpair_list[s3]) {
                          if (s4 > s4_max)
                              break;  // for each s3, s4 are stored in
                                      // monotonically increasing order
                                      
                          const auto Dnorm1234 =
                              do_schwartz_screen
                                  ? std::max(D_shblk_norm(s1, s4),
                                             std::max(D_shblk_norm(s2, s4),
                                                      Dnorm123))
                                  : 0.;

                          if (do_schwartz_screen &&
                              Dnorm1234 * Q12 * Schwartz(s3, s4) < k_precision)
                              continue;

                          auto bf4_first = shell2bf[s4];
                          auto n4 = obs[s4].size();

                          num_k_computed += n1 * n2 * n3 * n4;

                          // compute the permutational degeneracy (i.e. # of
                          // equivalents) of the given shell set
                          auto s12_deg = (s1 == s2) ? 1.0 : 2.0;
                          auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
                          auto s12_34_deg =
                              (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;
                          auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

#if defined(REPORT_INTEGRAL_TIMINGS)
                          timer.start(0);
#endif

                      const auto* buf =
                          engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);

#if defined(REPORT_INTEGRAL_TIMINGS)
                          timer.stop(0);
#endif

                          // 1) each shell set of integrals contributes up to 6
                          // shell sets of the Fock matrix:
                          //    F(a,b) += (ab|cd) * D(c,d)
                          //    F(c,d) += (ab|cd) * D(a,b)
                          //    F(b,d) -= 1/4 * (ab|cd) * D(a,c)
                          //    F(b,c) -= 1/4 * (ab|cd) * D(a,d)
                          //    F(a,c) -= 1/4 * (ab|cd) * D(b,d)
                          //    F(a,d) -= 1/4 * (ab|cd) * D(b,c)
                          // 2) each permutationally-unique integral (shell set)
                          // must be scaled by its degeneracy,
                          //    i.e. the number of the integrals/sets equivalent
                          //    to it
                          // 3) the end result must be symmetrized
                         for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                             const auto bf1 = f1 + bf1_first;
                             for (auto f2 = 0; f2 != n2; ++f2) {
                                 const auto bf2 = f2 + bf2_first;
                                 for (auto f3 = 0; f3 != n3; ++f3) {
                                     const auto bf3 = f3 + bf3_first;
                                     for (auto f4 = 0; f4 != n4;
                                          ++f4, ++f1234) {
                                         const auto bf4 = f4 + bf4_first;

                                         const auto value = buf[f1234];

                                         const auto value_scal_by_deg =
                                             value * s1234_deg;

                                         // g(bf1,bf2) += D(bf3,bf4) *
                                         // value_scal_by_deg;
                                         // g(bf3,bf4) += D(bf1,bf2) *
                                         // value_scal_by_deg;
                                         g(bf1, bf3) -= 0.25 * D(bf2, bf4) *
                                                        value_scal_by_deg;
                                         g(bf2, bf4) -= 0.25 * D(bf1, bf3) *
                                                        value_scal_by_deg;
                                         g(bf1, bf4) -= 0.25 * D(bf2, bf3) *
                                                        value_scal_by_deg;
                                         g(bf2, bf3) -= 0.25 * D(bf1, bf4) *
                                                        value_scal_by_deg;
                                     }
                                 }
                             }
                         }
                      }
                  }
              }
          }
      }

  };  // end of lambda

  namespace tim = std::chrono;
  using dur = tim::duration<double>;

  auto j0 = tim::high_resolution_clock::now();
  libint2::parallel_do(lambdaJ);
  auto j1 = tim::high_resolution_clock::now();
  libint2::parallel_do(lambdaK);
  auto k1 = tim::high_resolution_clock::now();

  dur jtime = j1 - j0;
  dur ktime = k1 - j1;

  std::cout << "\tJ time = " << jtime.count() << std::endl;
  std::cout << "\tPrescreen time = " << pstime.count() << std::endl;
  std::cout << "\tK time = " << ktime.count() << std::endl;
  k_times.push_back(ktime.count());

  // accumulate contributions from all threads
  for(size_t i=1; i!=nthreads; ++i) {
    G[0] += G[i];
  }

#if defined(REPORT_INTEGRAL_TIMINGS)
  double time_for_ints = 0.0;
  for(auto& t: timers) {
    time_for_ints += t.read(0);
  }
  std::cout << "time for integrals = " << time_for_ints << std::endl;
  for(int t=0; t!=nthreads; ++t)
    engines[t].print_timers();
#endif

  Matrix GG = 0.5 * (G[0] + G[0].transpose());

  std::cout << "\t# of coloumb integrals = " << num_j_computed << std::endl;
  std::cout << "\t# of exchange integrals = " << num_k_computed << std::endl;

  j_ints.push_back(num_j_computed);
  k_ints.push_back(num_k_computed);

  // symmetrize the result and return
  return GG;
}

Matrix compute_2body_fock_general(const BasisSet& obs,
                                  const Matrix& D,
                                  const BasisSet& D_bs,
                                  bool D_is_shelldiagonal,
                                  double precision) {

  const auto n = obs.nbf();
  const auto nshells = obs.size();
  const auto n_D = D_bs.nbf();
  assert(D.cols() == D.rows() && D.cols() == n_D);

  using libint2::nthreads;
  std::vector<Matrix> G(nthreads, Matrix::Zero(n,n));

  // construct the 2-electron repulsion integrals engine
  typedef libint2::TwoBodyEngine<libint2::Coulomb> coulomb_engine_type;
  std::vector<coulomb_engine_type> engines(nthreads);
  engines[0] = coulomb_engine_type(std::max(obs.max_nprim(),D_bs.max_nprim()),
                                   std::max(obs.max_l(), D_bs.max_l()), 0);
  engines[0].set_precision(precision); // shellset-dependent precision control will likely break positive definiteness
                                       // stick with this simple recipe
  for(size_t i=1; i!=nthreads; ++i) {
    engines[i] = engines[0];
  }
  auto shell2bf = obs.shell2bf();
  auto shell2bf_D = D_bs.shell2bf();

  auto lambda = [&] (int thread_id) {

    auto& engine = engines[thread_id];
    auto& g = G[thread_id];

    // loop over permutationally-unique set of shells
    for(auto s1=0l, s1234=0l; s1!=nshells; ++s1) {

      auto bf1_first = shell2bf[s1]; // first basis function in this shell
      auto n1 = obs[s1].size();   // number of basis functions in this shell

      for(auto s2=0; s2<=s1; ++s2) {

        auto bf2_first = shell2bf[s2];
        auto n2 = obs[s2].size();

        for(auto s3=0; s3<D_bs.size(); ++s3) {

          auto bf3_first = shell2bf_D[s3];
          auto n3 = D_bs[s3].size();

          auto s4_begin = D_is_shelldiagonal ? s3 : 0;
          auto s4_fence = D_is_shelldiagonal ? s3+1 : D_bs.size();

          for(auto s4=s4_begin; s4!=s4_fence; ++s4, ++s1234) {

            if (s1234 % nthreads != thread_id)
              continue;

            auto bf4_first = shell2bf_D[s4];
            auto n4 = D_bs[s4].size();

            // compute the permutational degeneracy (i.e. # of equivalents) of the given shell set
            auto s12_deg = (s1 == s2) ? 1.0 : 2.0;

            if (s3 >= s4) {
              auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
              auto s1234_deg = s12_deg * s34_deg;
              //auto s1234_deg = s12_deg;
              const auto* buf_J = engine.compute(obs[s1], obs[s2], D_bs[s3], D_bs[s4]);

              for(auto f1=0, f1234=0; f1!=n1; ++f1) {
                const auto bf1 = f1 + bf1_first;
                for(auto f2=0; f2!=n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for(auto f3=0; f3!=n3; ++f3) {
                    const auto bf3 = f3 + bf3_first;
                    for(auto f4=0; f4!=n4; ++f4, ++f1234) {
                      const auto bf4 = f4 + bf4_first;

                      const auto value = buf_J[f1234];
                      const auto value_scal_by_deg = value * s1234_deg;
                      g(bf1,bf2) += 2.0 * D(bf3,bf4) * value_scal_by_deg;
                    }
                  }
                }
              }
            }

            const auto* buf_K = engine.compute(obs[s1], D_bs[s3], obs[s2], D_bs[s4]);

            for(auto f1=0, f1324=0; f1!=n1; ++f1) {
              const auto bf1 = f1 + bf1_first;
              for(auto f3=0; f3!=n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for(auto f2=0; f2!=n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for(auto f4=0; f4!=n4; ++f4, ++f1324) {
                    const auto bf4 = f4 + bf4_first;

                    const auto value = buf_K[f1324];
                    const auto value_scal_by_deg = value * s12_deg;
                    g(bf1,bf2) -= D(bf3,bf4) * value_scal_by_deg;
                  }
                }
              }
            }

          }
        }
      }
    }

  }; // thread lambda

  libint2::parallel_do(lambda);

  // accumulate contributions from all threads
  for(size_t i=1; i!=nthreads; ++i) {
    G[0] += G[i];
  }

  // symmetrize the result and return
  return 0.5 * (G[0] + G[0].transpose());
}

#ifdef HAVE_DENSITY_FITTING

// uncomment if want to compute statistics of shell blocks
//#define COMPUTE_DF_INTS_STATS 1

Matrix
DFFockEngine::compute_2body_fock_dfC(const Matrix& Cocc) {

#ifdef _OPENMP
  const auto nthreads = omp_get_max_threads();
#else
  const auto nthreads = 1;
#endif

  const auto n          =  obs.nbf();
  const auto ndf        = dfbs.nbf();
#ifndef _OPENMP
  libint2::Timers<5> timer;
  timer.set_now_overhead(25);
#endif // not defined _OPENMP

  typedef btas::RangeNd<CblasRowMajor, std::array<long, 1> > Range1d;
  typedef btas::RangeNd<CblasRowMajor, std::array<long, 2> > Range2d;
  typedef btas::Tensor<double, Range1d> Tensor1d;
  typedef btas::Tensor<double, Range2d> Tensor2d;

  // using first time? compute 3-center ints and transform to inv sqrt repreentation
  if (xyK.size() == 0) {

    const auto nshells    =  obs.size();
    const auto nshells_df = dfbs.size();
    const auto unitshell = libint2::Shell::unit();

    // construct the 2-electron 3-center repulsion integrals engine
    typedef libint2::TwoBodyEngine<libint2::Coulomb> coulomb_engine_type;
    std::vector<coulomb_engine_type> engines(nthreads);
    engines[0] = coulomb_engine_type(std::max(obs.max_nprim(), dfbs.max_nprim()),
                                     std::max(obs.max_l(), dfbs.max_l()), 0);
    for(size_t i=1; i!=nthreads; ++i) {
      engines[i] = engines[0];
    }

    auto shell2bf    =  obs.shell2bf();
    auto shell2bf_df = dfbs.shell2bf();

    Tensor3d xyZ{n, n, ndf};

#if COMPUTE_DF_INTS_STATS
    const double count_threshold = 1e-7;
    std::atomic<size_t> nints{0};  // number of ints in shell blocks with Frobenius norm > count_threshold
    std::atomic<size_t> nints2{0}; // number of ints in shell blocks with Frobenius norm/elem > count_threshold
    typedef btas::Tensor<char, Range3d> Tensor3dBool;
    const size_t nobs_per_mol = 24; // cc-pVDZ
    const size_t ndfbs_per_mol = 84; // cc-pVDZ-RI
    const size_t nmols = n/nobs_per_mol;
    Tensor3dBool nonzero_mol_blocks{nmols, nmols, nmols}; // number of molecule blocks that contain shell blocks with Frobenius norm > count_threshold
    Tensor3d mol_blocks_frob2{nmols, nmols, nmols}; // sum of Frobenius norms squared in each molecule block
    std::fill(nonzero_mol_blocks.begin(), nonzero_mol_blocks.end(), 0);
    std::fill(mol_blocks_frob2.begin(), mol_blocks_frob2.end(), 0.);
#endif // COMPUTE_DF_INTS_STATS

  #ifdef _OPENMP
    #pragma omp parallel
  #endif
    {
  #ifdef _OPENMP
      auto thread_id = omp_get_thread_num();
  #else
      auto thread_id = 0;
  #endif

      // loop over permutationally-unique set of shells
      for(auto s1=0l, s123=0l; s1!=nshells; ++s1) {

        auto bf1_first = shell2bf[s1]; // first basis function in this shell
        auto n1 = obs[s1].size();// number of basis functions in this shell

        for(auto s2=0; s2!=nshells; ++s2) {

          auto bf2_first = shell2bf[s2];
          auto n2 = obs[s2].size();
          const auto n12 = n1*n2;

          for(auto s3=0; s3!=nshells_df; ++s3, ++s123) {

            if (s123 % nthreads != thread_id)
              continue;

            auto bf3_first = shell2bf_df[s3];
            auto n3 = dfbs[s3].size();
            const auto n123 = n12*n3;

  #ifndef _OPENMP
            timer.start(0);
  #endif

            const auto* buf = engines[thread_id].compute(obs[s1], obs[s2], dfbs[s3], unitshell);

  #ifndef _OPENMP
            timer.stop(0);
  #endif

#if COMPUTE_DF_INTS_STATS
            const auto buf_2norm = sqrt(std::inner_product(buf, buf+n123, buf, 0.));
            const auto buf_2norm_scaled = sqrt(buf_2norm * buf_2norm/ n123);
            const auto buf_infnorm = std::abs(
                  *std::max_element(buf, buf+n123, [](double a, double b){return std::abs(a) < std::abs(b);})
            );
            if (buf_2norm > count_threshold) {
              nints += n123;

              nonzero_mol_blocks(bf1_first/nobs_per_mol,
                                 bf2_first/nobs_per_mol,
                                 bf3_first/ndfbs_per_mol) = 1;
            }
            if (buf_2norm_scaled > count_threshold) {
              nints2 += n123;
            }
            mol_blocks_frob2(bf1_first/nobs_per_mol,
                             bf2_first/nobs_per_mol,
                             bf3_first/ndfbs_per_mol) += buf_2norm*buf_2norm;
#endif // COMPUTE_DF_INTS_STATS

  #ifndef _OPENMP
            timer.start(1);
  #endif

            auto lower_bound = {bf1_first, bf2_first, bf3_first};
            auto upper_bound = {bf1_first+n1, bf2_first+n2, bf3_first+n3};
            auto view = btas::make_view( xyZ.range().slice(lower_bound, upper_bound),
                                         xyZ.storage());
            std::copy(buf, buf+n123, view.begin());

  #ifndef _OPENMP
            timer.stop(1);
  #endif

          } // s3
        } // s2
      } // s1

    } // omp parallel

  #ifndef _OPENMP
    std::cout << "time for integrals = " << timer.read(0) << std::endl;
    std::cout << "time for copying into BTAS = " << timer.read(1) << std::endl;
    engines[0].print_timers();
  #endif // not defined _OPENMP

#if COMPUTE_DF_INTS_STATS
    {
      const auto nints_per_molblock = nobs_per_mol * nobs_per_mol * ndfbs_per_mol;
      const size_t nints_mols = std::count(nonzero_mol_blocks.begin(), nonzero_mol_blocks.end(), 1) *
                                nints_per_molblock;
      const size_t nints_mols2= std::count_if(mol_blocks_frob2.begin(), mol_blocks_frob2.end(),
                                              [&](double a) { return sqrt(a) > count_threshold; }
                                             ) *
                                                 nints_per_molblock;
      const size_t nints_mols3= std::count_if(mol_blocks_frob2.begin(), mol_blocks_frob2.end(),
                                              [&](double a) { return sqrt(a)/nints_per_molblock > count_threshold; }
                                             ) *
                                                 nints_per_molblock;
      std::cout << "# of ints in shell blocks with norm greater than " << count_threshold << " = " << nints << std::endl;
      std::cout << "# of ints in shell blocks with scaled norm greater than " << count_threshold << " = " << nints2 << std::endl;
      std::cout << "# of ints in molecule blocks whose any shell-block norm was greater than " << count_threshold << " = " << nints_mols << std::endl;
      std::cout << "# of ints in molecule blocks with norm greater than " << count_threshold << " = " << nints_mols2 << std::endl;
      std::cout << "# of ints in molecule blocks with scaled norm greater than " << count_threshold << " = " << nints_mols3 << std::endl;
      std::cout << "# of total ints = " << n*n*ndf << std::endl;
    }
#endif // COMPUTE_DF_INTS_STATS

    timer.start(2);

    Matrix V = compute_2body_2index_ints(dfbs);
    Eigen::LLT<Matrix> V_LLt(V);
    Matrix I = Matrix::Identity(ndf, ndf);
    auto L = V_LLt.matrixL();
    Matrix V_L = L;
    Matrix Linv = L.solve(I).transpose();
    // check
  //  std::cout << "||V - L L^t|| = " << (V - V_L * V_L.transpose()).norm() << std::endl;
  //  std::cout << "||I - L L^-1^t|| = " << (I - V_L * Linv.transpose()).norm() << std::endl;
  //  std::cout << "||V^-1 - L^-1 L^-1^t|| = " << (V.inverse() - Linv * Linv.transpose()).norm() << std::endl;

    Tensor2d K{ndf, ndf};
    std::copy(Linv.data(), Linv.data()+ndf*ndf, K.begin());

    xyK = Tensor3d{n, n, ndf};
    btas::contract(1.0, xyZ, {1,2,3}, K, {3,4}, 0.0, xyK, {1,2,4});
    xyZ = Tensor3d{0,0,0}; // release memory

    timer.stop(2);
    std::cout << "time for integrals tform = " << timer.read(2) << std::endl;
  } // if (xyK.size() == 0)

  // compute exchange
  timer.start(3);

  const auto nocc = Cocc.cols();
  Tensor2d Co{n, nocc};
  std::copy(Cocc.data(), Cocc.data()+n*nocc, Co.begin());
  Tensor3d xiK{n, nocc, ndf};
  btas::contract(1.0, xyK, {1,2,3}, Co, {2,4}, 0.0, xiK, {1,4,3});

  Tensor2d G{n, n};
  btas::contract(1.0, xiK, {1,2,3}, xiK, {4,2,3}, 0.0, G, {1,4});

  timer.stop(3);
  std::cout << "time for exchange = " << timer.read(3) << std::endl;

  // compute Coulomb
  timer.start(4);

  Tensor1d Jtmp{ndf};
  btas::contract(1.0, xiK, {1,2,3}, Co, {1,2}, 0.0, Jtmp, {3});
  xiK = Tensor3d{0,0,0};
  btas::contract(2.0, xyK, {1,2,3}, Jtmp, {3}, -1.0, G, {1,2});

  timer.stop(4);
  std::cout << "time for coulomb = " << timer.read(4) << std::endl;

  // copy result to an Eigen::Matrix
  Matrix result(n, n);
  std::copy(G.cbegin(), G.cend(), result.data());
  return result;
}
#endif // HAVE_DENSITY_FITTING
