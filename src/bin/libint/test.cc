
#include <iostream>
#include <vector>
#include <rr.h>
#include <dg.h>
#include <typelist.h>
#include <integral.h>
#include <iter.h>
#include <policy_spec.h>
#include <intset_to_ints.h>
#include <strategy.h>

using namespace std;
using namespace libint2;

static int try_main (int argc, char* argv[]);
static void test_typelists();
static void test_old_integrals();
static void test_integrals();

int main (int argc, char* argv[])
{

  try {
    try_main(argc,argv);
  }
  catch(std::exception& a) {
    cout << endl
         << "  WARNING! Caught a standard exception:" << endl
         << "    " << a.what() << endl << endl;
  }
}

int try_main (int argc, char* argv[])
{

  unsigned int am[][1] = { {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}};
  CGShell sh_s(am[0]);
  CGShell sh_p(am[1]);
  CGShell sh_d(am[2]);
  CGShell sh_f(am[3]);
  CGShell sh_g(am[4]);
  CGShell sh_h(am[5]);
  CGShell sh_i(am[6]);
  CGShell sh_k(am[7]);
  CGShell sh_l(am[8]);
  CGShell sh_m(am[9]);
  CGShell sh_n(am[10]);

  vector<CGShell> sh_12[TwoERep::Properties::np];
  sh_12[0].push_back(sh_p);
  sh_12[1].push_back(sh_p);
  vector<CGShell> sh_34[TwoERep::Properties::np];
  sh_34[0].push_back(sh_p);
  sh_34[1].push_back(sh_p);

#if 0
  test_typelists();
#endif
  
  typedef TwoPRep_11_11<CGShell> TwoPRep_sh_11_11;
  SafePtr<TwoPRep_sh_11_11> pppp_quartet = TwoPRep_sh_11_11::Instance(sh_p,sh_p,sh_p,sh_p,0);
  pppp_quartet->print(cout);

  // Create a DAG for a HRR+VRR case
  DirectedGraph dg_xxxx;

  SafePtr<DGVertex> pppp_ptr = dynamic_pointer_cast<DGVertex,TwoPRep_sh_11_11>(pppp_quartet);
#if 1
  dg_xxxx.append_target(pppp_ptr);
  dg_xxxx.apply_to_all< HRR_ab_11_TwoPRep_11_sh >();
  dg_xxxx.apply_to_all< HRR_cd_11_TwoPRep_11_sh >();
  dg_xxxx.apply_to_all< VRR_a_11_TwoPRep_11_sh >();
  dg_xxxx.apply_to_all< VRR_c_11_TwoPRep_11_sh >();
  dg_xxxx.apply_to_all< VRR_b_11_TwoPRep_11_sh >();
  dg_xxxx.apply_to_all< VRR_d_11_TwoPRep_11_sh >();

  dg_xxxx.traverse();
  dg_xxxx.debug_print_traversal(cout);
  dg_xxxx.reset();
#endif

  // test iterators
  SafePtr<CGShell> sh_ptr(new CGShell(sh_i));
  SubIteratorBase<CGShell> siter1(sh_ptr);
  for(siter1.init(); siter1; ++siter1)
    siter1.elem()->print(cout);

  SafePtr<TwoPRep_sh_11_11> obj(pppp_quartet);
  obj->print(cout);
  SubIteratorBase< TwoPRep_sh_11_11 > siter2(obj);
  for(siter2.init(); siter2; ++siter2)
    siter2.elem()->print(cout);

  // Create a DAG for a shell-quartet-to-ints case
  DirectedGraph dg_xxxx2;

  dg_xxxx2.append_target(pppp_ptr);
  dg_xxxx2.apply_to_all< IntegralSet_to_Integrals<TwoPRep_sh_11_11 > >();

  dg_xxxx2.traverse();
  dg_xxxx2.debug_print_traversal(cout);
  dg_xxxx2.reset();

  SafePtr<DirectedGraph> dg_xxxx3(new DirectedGraph);
  SafePtr<Strategy> strat(new Strategy);
  SafePtr<TwoPRep_sh_11_11> xsxs_quartet = TwoPRep_sh_11_11::Instance(sh_p,sh_s,sh_p,sh_s,0);
  xsxs_quartet->print(cout);
  SafePtr<DGVertex> xsxs_ptr = dynamic_pointer_cast<DGVertex,TwoPRep_sh_11_11>(xsxs_quartet);
  dg_xxxx3->append_target(xsxs_ptr);
  dg_xxxx3->apply(strat);
  dg_xxxx3->optimize_rr_out();
  dg_xxxx3->traverse();
  dg_xxxx3->debug_print_traversal(cout);
  cout << "The number of vertices = " << dg_xxxx3->num_vertices() << endl;
  dg_xxxx3->reset();

}

#if 0
void
test_typelists()
{
  typedef PTYPELIST_2( Tag2, CGShell, CGShell) two_shells;
  two_shells xx;
  typedef PTYPELIST_3( Tag2, CGShell, CGShell, CGShell) three_shells;
  three_shells xxx;
  typedef GenIntegral<TwoERep,two_shells,two_shells> NewTwoERep_2b2k;
  NewTwoERep_2b2k xxxx(xx,xx);
  
  ValueAt<two_shells, 1>::ResultType xx_1 = ValueAt<two_shells,1>::typelist_member(xx);

  cout << "Length of my_typelist = " << Length<two_shells>::value << endl;
  CGShell xx_2(xx_1);
  xx_2.inc();
}
#endif
