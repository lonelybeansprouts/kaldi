#include <fst/fstlib.h>
#include <fst/fst-decl.h>


// see pre-determinize.h for documentation.
template<class Arc> void AddIsymsSelfLoops(MutableFst<Arc> *fst, std::vector<typename Arc::Label> &isyms,
                                      std::vector<typename Arc::Label> &disambig_syms) {
  assert(fst != NULL);
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  std::set<Label> isyms_set, disambi_set;
  for (size_t i = 0; i < isyms.size(); i++) {
    isyms_set.insert(isyms[i]);
  }
  for (size_t i = 0; i < disambig_syms.size(); i++){
      disambi_set.insert(disambig_syms[i]);
  }

  for (StateIterator<MutableFst<Arc> > siter(*fst); ! siter.Done(); siter.Next()) {
    StateId state = siter.Value();
    bool this_state_needs_self_loops = false; //= (fst->Final(state) != Weight::Zero());
    for (ArcIterator<MutableFst<Arc> > aiter(*fst, state); ! aiter.Done(); aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (isyms_set.count(arc.ilabel)!=0 && disambig_syms.count(arc.ilabel)==0 && state!=arc.nextstate){
        this_state_needs_self_loops = true;
      }
    }
    if (this_state_needs_self_loops) {
      Arc arc_add;
      arc_add.ilabel = isyms[i];
      arc_add.olabel = 0;
      arc_add.weight = Weight::One();
      arc_add.nextstate = arc.nextstate;
      fst->AddArc(arc.nextstate, arc_add);
    }
  }
}


int main(int args, char *argv[]){


  
}

