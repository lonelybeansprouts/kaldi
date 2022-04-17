

// bin/decode-faster-mapped.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "fstext/pre-determinize.h"
#include "decoder/faster-decoder.h"
#include "decoder/decodable-matrix.h"
#include "base/timer.h"
#include "lat/kaldi-lattice.h" // for {Compact}LatticeArc

#include "fst/fstlib.h"
#include "fstext/table-matcher.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"

#include <chrono>
#include <codecvt>
#include <locale>
#include <regex>

using namespace kaldi;
typedef kaldi::int32 int32;
using fst::SymbolTable;
using fst::VectorFst;
using fst::Fst;
using fst::StdArc;
using namespace fst;


//#if !defined(__ANDROID__) && !defined(__IOS__)
long GetTime() {
#ifdef _WIN32
  return clock();
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
#endif
}
//#endif


std::shared_ptr<fst::StdVectorFst> ConstructLexiconFst(
    std::unordered_map<int, std::vector<std::vector<int>>> &lexicon, int &sil_id, int &boundary_id) {
  std::shared_ptr<fst::StdVectorFst> ifst = std::make_shared<fst::StdVectorFst>();
  ifst->DeleteStates();
  int start_state = ifst->AddState();
  ifst->SetStart(start_state);
  int end_state = ifst->AddState();
  ifst->AddArc(start_state, fst::StdArc(sil_id, 0, 0.0, start_state));
  ifst->AddArc(end_state, fst::StdArc(sil_id, 0, 0.0, end_state));

  for (auto &item : lexicon){
    int word_id = item->first;
    auto &phone_seqs = item->second;
    for (auto &phone_seq : phone_seqs){
      int cur_phone_state = start_state;
      int next_phone_state;
      for (auto &phone : phone_seq){
        next_phone_state = ifst->AddState();
        ifst->AddArc(cur_phone_state, fst::StdArc(phone, 0, 0.0, next_phone_state));
        ifst->AddArc(next_phone_state, fst::StdArc(phone, 0, 0.0, next_phone_state)); //self loop
        cur_phone_state = next_phone_state;
        next_phone_state = ifst->AddState();
        ifst->AddArc(cur_phone_state, fst::StdArc(boundary_id, 0, 0.0, next_phone_state));
        cur_phone_state = next_phone_state;
      }
      ifst->AddArc(cur_phone_state, fst::StdArc(0, word_id, 0.0, end_state));
    }
  }
  ifst->AddArc(end_state, fst::StdArc(0, 0, 0.0, start_state));  //loop
  final_state = ifst->AddState();
  ifst->AddArc(end_state, fst::StdArc(0, 0, 0.0, final_state));
  ifst->SetFinal(final_state, fst::StdArc::Weight::One());
  fst::ArcSort(ifst.get(), fst::StdILabelCompare());
  return ifst;
}




int main(int argc, char *argv[]) {


  const char *usage = "\n";
  ParseOptions po(usage);
  po.Read(argc, argv);

  std::string phone_syms_path = po.GetArg(1);
  std::string word_syms_path = po.GetArg(2);
  std::string lexicon_path = po.GetArg(3);
  std::string fst_in_filename = po.GetArg(4);
  std::string fst_out_filename = po.GetOptArg(5);

  fst::SymbolTable *phone_syms = NULL;
  if (phone_syms_path != "") {
    phone_syms = fst::SymbolTable::ReadText(phone_syms_path);
    if (!phone_syms)
      KALDI_ERR << "Could not read symbol table from file "<<phone_syms_path;
  }

  fst::SymbolTable *word_syms = NULL;
  if (word_syms_path != "") {
    word_syms = fst::SymbolTable::ReadText(word_syms_path);
    if (!word_syms)
      KALDI_ERR << "Could not read symbol table from file "<<word_syms_path;
  }

  int sil_id = phone_syms->Find("sil");
  assert(sil_id > 0);
  int boundary_id = phone_syms->Find("#");
  assert(boundary_id > 0);
  int dis_ambig_id = word_syms->Find("#0")
  assert(dis_ambig_id > 0);

  std::unordered_map<int, std::vector<std::vector<int>>> lexicon;{
    std::ifstream ifs(lexicon_path);
    std::string line;
    while (std::getline(ifs, line)){
      std::stringstream ss(line);
      std::string word, phone;
      int word_id, phone_id;
      std::vector<int> phoneseq;
      ss >> word;
      word_id = word_syms->Find(word);
      assert(word_id >= 0);
      while (ss >> phone){
        phone_id = phone_syms->Find(phone);
        assert(phone_id>=0);
        phoneseq.emplace_back(phone_id);
      }
      if (lexicon.count(word_id)>0){
        auto &phoneseq_vec = lexicon[word_id];
        phoneseq_vec.emplace_back(phoneseq);
      }else{
        lexicon[word_id] = {phoneseq};
      }
    }
  }

  fst::VectorFst<fst::StdArc> *gfst =
        fst::VectorFst<fst::StdArc>::Read(fst_in_filename);


  auto lfst = ConstructLexiconFst(lexicon, sil_id, boundary_id);

  AddSelfLoops(lfst.get(), {0}, {dis_ambig_id});



  fst::VectorFst<fst::StdArc> cfst;{
    TableComposeOptions opts;
    std::string match_side = "right";
    std::string compose_filter = "sequence";

    if (match_side == "left") {
      opts.table_match_type = MATCH_OUTPUT;
    } else if (match_side == "right") {
      opts.table_match_type = MATCH_INPUT;
    } else {
      KALDI_ERR << "Invalid match-side option: " << match_side;
    }
    if (compose_filter == "alt_sequence") {
      opts.filter_type = ALT_SEQUENCE_FILTER;
    } else if (compose_filter == "auto") {
      opts.filter_type = AUTO_FILTER;
    } else  if (compose_filter == "match") {
      opts.filter_type = MATCH_FILTER;
    } else  if (compose_filter == "sequence") {
      opts.filter_type = SEQUENCE_FILTER;
    } else {
      KALDI_ERR << "Invalid compose-filter option: " << compose_filter;
    }

    TableCompose(*lfst, *gfst, &cfst, opts);
    // fst::Compose(*lfst, *rfst, &cfst);
  }

  cfst.Write(fst_out_filename);

  delete gfst;
  delete phone_syms;
  delete word_syms;

}


