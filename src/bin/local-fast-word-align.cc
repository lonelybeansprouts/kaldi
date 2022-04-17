

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
#include <cmath>

using namespace kaldi;
typedef kaldi::int32 int32;
using fst::SymbolTable;
using fst::VectorFst;
using fst::Fst;
using fst::StdArc;
using namespace fst;


struct WordAlignmentInfo {
  int32 word_id;
  int32 start_frame;
  int32 length_in_frames;
  double confidence;
};

struct PhoneAlignmentInfo {
  int32 phone_id;
  int32 start_frame;
  int32 length_in_frames;
  double confidence;
};


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


std::string TextSplit(const std::string& str) {
  std::wstring_convert<std::codecvt_utf8<wchar_t> > strCnv;
  std::wstring ws = strCnv.from_bytes(str);
  std::wregex re(L"([\u4e00-\u9fa5\u0800-\u4e00\uac00-\ud7ff，。！；？])");
  // insert blank before and after chinese 、japanese and korea tokens
  auto wt = std::regex_replace(ws, re, L" $1 ");
  return strCnv.to_bytes(wt);
}

std::shared_ptr<fst::StdVectorFst> ConstructLeftFst(const Matrix<BaseFloat> &loglikes, const float &threshold, const int &min_phone_num, const std::set<int> &phone_in_vision) {
  //topk=[1,classes]
  auto get_topk = [](const Matrix<BaseFloat> &loglikes, const std::set<int> &phone_in_vision, const int &row, const float &threshold, const int &min_phone_num, std::vector<float> &tmp_values){
    int cols = loglikes.NumCols();
    tmp_values.clear();
    // std::cout<<"\n\n";

    for (int i=0; i<cols; i++ ){
      if (phone_in_vision.count(i)>0){
        // std::cout<<loglikes(row,i)<< " ";
        tmp_values.emplace_back(exp(loglikes(row,i)));
      }
    }
    // std::cout<<"\n\n";
    std::sort(tmp_values.begin(), tmp_values.end(), [](const float &a, const float &b){return a>b;});
    float sum = 0;
    int count = 0;
    for (auto &x : tmp_values){
      sum += x;
      count++;
      if (sum>=threshold && count>=min_phone_num){
        return log(x);
      }
    }
    return log(tmp_values.back());
  };
  std::shared_ptr<fst::StdVectorFst> ifst = std::make_shared<fst::StdVectorFst>();
  static std::vector<float> tmp_values;
  ifst->DeleteStates();
  int cur_state = ifst->AddState();
  int next_state;
  ifst->SetStart(cur_state);
  for (int row=0; row<loglikes.NumRows(); row++) {
    float sum = 0.0;
    next_state = ifst->AddState();
    float threshold_logp = get_topk(loglikes, phone_in_vision, row, threshold, min_phone_num, tmp_values);

    // std::cout<<"threshold_logp: "<<threshold_logp<<"\n";

    for (int col=0; col<loglikes.NumCols(); col++) {
      if (col==0 || phone_in_vision.count(col)<=0) continue;  //ignore eps and oov label
      // std::cout<<loglikes(row, col)<<" ";
      if (-loglikes(row, col) <= -threshold_logp+(1e-6)){
        ifst->AddArc(cur_state,fst::StdArc(col, col, -loglikes(row, col), next_state));
        // std::cout<<exp(loglikes(row, col))<<" ";
        sum += exp(loglikes(row, col));
      }
    }
    cur_state = next_state;
    // std::cout<<"\n";
    // std::cout<<"sum: "<<sum<<"\n";
  }
  ifst->SetFinal(next_state, fst::StdArc::Weight::One());
  fst::ArcSort(ifst.get(), fst::StdOLabelCompare());  //the fst is olabel sorted already
  return ifst;
}


std::shared_ptr<fst::StdVectorFst> ConstructRightFst(
    std::unordered_map<int, std::vector<std::vector<int>>> &lexicon, std::vector<int> &outsyms, int &sil_id, int &boundary_id, std::set<int> &phone_in_vision) {
  phone_in_vision.clear();
  phone_in_vision.insert(sil_id);
  phone_in_vision.insert(boundary_id);
  std::shared_ptr<fst::StdVectorFst> ifst = std::make_shared<fst::StdVectorFst>();
  ifst->DeleteStates();
  int cur_state = ifst->AddState();
  int next_state;
  ifst->SetStart(cur_state);

  //self loop for sils
  //process all o label
  for (int i=0; i<outsyms.size(); i++){
    auto &cur_outsym = outsyms[i];
    ifst->AddArc(cur_state, fst::StdArc(sil_id, -sil_id, 0.0, cur_state));
    next_state = ifst->AddState();
    auto &phone_seqs = lexicon[cur_outsym];
    // for (auto &phone_seq : phone_seqs){
    for (int j=0; j<phone_seqs.size(); j++){
      auto &phone_seq = phone_seqs[j];
      int cur_phone_state = cur_state;
      int next_phone_state;
      // for (auto &phone : phone_seq){
      for (int k=0; k<phone_seq.size(); k++){
        auto &phone = phone_seq[k];
        phone_in_vision.emplace(phone);
        if (k == phone_seq.size()-1){ //last phone
          ifst->AddArc(cur_phone_state, fst::StdArc(phone, -phone, 0.0, next_state));
          ifst->AddArc(next_state, fst::StdArc(phone, -phone, 0.0, next_state)); //self loop
        }else{
          next_phone_state = ifst->AddState();
          ifst->AddArc(cur_phone_state, fst::StdArc(phone, -phone, 0.0, next_phone_state));
          ifst->AddArc(next_phone_state, fst::StdArc(phone, -phone, 0.0, next_phone_state)); //self loop
          cur_phone_state = next_phone_state;
          next_phone_state = ifst->AddState();
          ifst->AddArc(cur_phone_state, fst::StdArc(boundary_id, -boundary_id, 0.0, next_phone_state));
          cur_phone_state = next_phone_state;
        }

      }
      if (j == phone_seqs.size()-1){
        cur_phone_state = next_state;
        next_state = ifst->AddState();
        ifst->AddArc(cur_phone_state, fst::StdArc(boundary_id, cur_outsym, 0.0, next_state));
      }
    }
    cur_state = next_state;
    if (i == outsyms.size()-1){
      ifst->AddArc(cur_state, fst::StdArc(sil_id, -sil_id, 0.0, cur_state));
    }
  }
  next_state = ifst->AddState();
  ifst->AddArc(cur_state, fst::StdArc(0, 0, 0.0, next_state));
  ifst->SetFinal(next_state, fst::StdArc::Weight::One());
  fst::ArcSort(ifst.get(), fst::StdILabelCompare());
  return ifst;
}




int main(int argc, char *argv[]) {


  const char *usage = "\n";
  ParseOptions po(usage);
  int topk = -1;
  po.Register("topk", &topk, "Write output in binary mode");
  po.Read(argc, argv);



  std::string loglikes_rspecifier = po.GetArg(1),
      groundtruth = po.GetArg(2),
      phone_syms_path = po.GetArg(3),
      word_syms_path = po.GetArg(4),
      lexicon_path = po.GetArg(5),
      ali_output_path = po.GetArg(6);


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

  std::unordered_map<std::string,std::vector<int>> groudtruth_map;{
    std::ifstream ifs(groundtruth);
    std::string line;
    int unk_word_id = word_syms->Find("<UNK>");
    assert(unk_word_id>0);
    while (std::getline(ifs, line)){
      int pos = line.find_first_of(" \t");
      assert(pos>0);
      std::string key(line.begin(), line.begin()+pos);
      std::string text(line.begin()+pos+1, line.end());
      text = TextSplit(text);
      std::stringstream ss(text);
      std::vector<int> text_ids;
      std::string word;
      while (ss >> word){
        int id = word_syms->Find(word);
        if (id > 0){
          text_ids.emplace_back(id);
        }else{
          std::cout<<"oov:"<<word<<"\n";
          text_ids.emplace_back(unk_word_id);
        }
      }
      groudtruth_map.emplace(key, text_ids);
    }
  }


  SequentialBaseFloatMatrixReader loglikes_reader(loglikes_rspecifier);
  int num_fail = 0;
  std::shared_ptr<std::ofstream> ofs;
  if (ali_output_path != "-"){
    ofs = std::make_shared<std::ofstream>(ali_output_path);
  }

  std::set<int> phone_in_vision;

  for (; !loglikes_reader.Done(); loglikes_reader.Next()) {
    std::string key = loglikes_reader.Key();
    const Matrix<BaseFloat> &loglikes (loglikes_reader.Value());
    if (loglikes.NumRows() == 0) {
      KALDI_WARN << "Zero-length utterance: " << key;
      num_fail++;
      continue;
    }
    if (groudtruth_map.count(key)<=0){
      std::cout<<key <<" is oov\n";
      assert(false);
    }
    auto &word_seqs = groudtruth_map[key];

    // topk = topk<0 ? loglikes.NumCols() : (topk>loglikes.NumCols()? loglikes.NumCols() : topk); //topk=[1,classes]


    // topk = 20;
    float threshold = 0.9;
    int min_phone_num = 10;

    auto start_time_2 = std::chrono::system_clock::now();
    auto rfst = ConstructRightFst(lexicon, word_seqs, sil_id, boundary_id, phone_in_vision);

    // for (auto &x : phone_in_vision){
    //   std::cout<<x<<" ";
    // }std::cout<<"\n";



    auto end_time_2 = std::chrono::system_clock::now();
    float time_cost2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_2-start_time_2).count()/1000.0;

    auto start_time_1 = std::chrono::system_clock::now();
    auto lfst = ConstructLeftFst(loglikes, threshold, min_phone_num, phone_in_vision);
    auto end_time_1 = std::chrono::system_clock::now();
    float time_cost1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_1-start_time_1).count()/1000.0;



    auto start_time_3 = std::chrono::system_clock::now();
    fst::VectorFst<fst::StdArc> cfst, onebest;
    {
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

      TableCompose(*lfst, *rfst, &cfst, opts);
      // fst::Compose(*lfst, *rfst, &cfst);
    }
    auto end_time_3 = std::chrono::system_clock::now();
    float time_cost3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_3-start_time_3).count()/1000.0;


    auto start_time_4 = std::chrono::system_clock::now();
    fst::ShortestPath(cfst, &onebest);
    auto end_time_4 = std::chrono::system_clock::now();
    float time_cost4 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_4-start_time_4).count()/1000.0;


    auto start_time_5 = std::chrono::system_clock::now();
    std::vector<fst::StdArc::Label> olabels;
    auto s = onebest.Start();
    if (s == fst::kNoStateId) {
      std::cerr << "StringFstToOutputLabels: Invalid start state" << std::endl;
      return false;
    }
    while (onebest.Final(s) == fst::StdArc::Weight::Zero()) {
      fst::ArcIterator<fst::Fst<fst::StdArc> > aiter(onebest, s);
      if (aiter.Done()) {
        std::cerr << "StringFstToOutputLabels: Does not reach final state" << std::endl;
        return false;
      }
      const fst::StdArc &arc = aiter.Value();
      olabels.push_back(arc.olabel);
      s = arc.nextstate;
      aiter.Next();
      if (!aiter.Done()) {
        std::cerr << "StringFstToOutputLabels: State has multiple outgoing arcs" << std::endl;
        return false;
      }
    }

    // for (auto &x : olabels){
    //   std::cout<<x <<" ";
    // }
    // std::cout<<"\n";

    WordAlignmentInfo word_info{-1,-1,0,-1};
    std::vector<WordAlignmentInfo> word_alignment;
    int frame_id = -1;
    for (int i=0; i<olabels.size(); i++){
      auto &olabel = olabels[i];
      if (olabel<0 || olabel>0) frame_id++; //that olabel<0 or olabel>0 means phone label and also means a frame
      if ((olabel<0 && -olabel!=sil_id) || (olabel>0)){
        if (word_info.start_frame < 0){
          word_info.start_frame = frame_id;
        }
        word_info.length_in_frames++;
      }
      if (olabel>0){
          word_info.word_id = olabel;
          assert(word_info.word_id>0 && word_info.start_frame>=0 && word_info.length_in_frames>0);
          word_alignment.push_back(word_info); //need copy
          word_info = {-1,-1,0,-1};
      }

    }
    auto end_time_5 = std::chrono::system_clock::now();
    float time_cost5 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_5-start_time_5).count()/1000.0;


    float time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_5-start_time_1).count()/1000.0;;
    float total_time=loglikes.NumRows()*10/1000.0;
    float rtf=time_cost/total_time;

    int frame_shift=10;
    int shift_delta=15;

    string cost_info;

    cost_info=cost_info+std::to_string(time_cost1)
              +"_"+std::to_string(time_cost2)
              +"_"+std::to_string(time_cost3)
              +"_"+std::to_string(time_cost4)
              +"_"+std::to_string(time_cost5)
              +"_"+std::to_string(time_cost)
              +"_"+std::to_string(total_time)
              +"_"+std::to_string(rtf);

    std::stringstream ss;
    ss<<key<<" "<<cost_info;
    for (auto &x : word_alignment){
      ss<<" "<<"("<<word_syms->Find(x.word_id)
                <<" "<< (x.start_frame==0 ? 0 :(x.start_frame*frame_shift+shift_delta)/1000.0)
                <<" "<<((x.start_frame+x.length_in_frames)*frame_shift+shift_delta)/1000.0<<")";
    }
    ss<<"\n";
    if (ofs != NULL) {
      *ofs << ss.str();
    }
    std::cout<<ss.str();



//  for (auto &x : word_alignment){
//    std::cout<<x.word_id<< " "<<x.start_frame<<" "<<x.length_in_frames<<"\n";
//  }

    // PhoneAlignmentInfo phone_info{-1, -1, 0, -1};
    // frame_id = -1;
    // for (int i=0; i<olabels.size(); i++){
    //   auto &olabel = olabels[i];
    //   if (olabel==0) continue;
    //   if (olabel < 0) frame_id++;//that olabel < 0 means phone label and also means a frame
    //   if (olabel<0) {
    //     if (sil_syms.count(-olabel))
    //       continue;
    //     if (phone_info.phone_id < 0)
    //       phone_info.phone_id = -olabel;
    //     if (phone_info.start_frame < 0)
    //       phone_info.start_frame = frame_id;
    //     if (phone_info.phone_id > 0)
    //       phone_info.length_in_frames++;
    //   }
    //   if (phone_info.phone_id>0&&-olabel!=phone_info.phone_id || olabel>0){ //meet phone end boundary
    //     assert(phone_info.phone_id>0 && phone_info.start_frame>=0 && phone_info.length_in_frames>0);
    //     phone_alignment.push_back(phone_info);
    //     phone_info = {-1, -1 , 0, -1};
    //   }

    //   if (i==olabels.size()-1 && phone_info.phone_id>0){
    //     phone_alignment.push_back(phone_info);
    //   }
    // }
  }
  if (ali_output_path != "-"){
    ofs->close();
  }
}


