// bin/compile-train-graphs.cc

// Copyright 2009-2012  Microsoft Corporation
//           2012-2015  Johns Hopkins University (Author: Daniel Povey)

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
#include "decoder/training-graph-compiler.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/decodable-matrix.h"
#include "base/timer.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "lat/word-align-lattice-lexicon.h"
#include "lat/lattice-functions.h"
#include "lat/lattice-functions-transition-model.h"

using namespace kaldi;
typedef kaldi::int32 int32;
using fst::SymbolTable;
using fst::VectorFst;
using fst::StdArc;
using fst::Fst;


void lat_one_best(CompactLattice &clat, CompactLattice &best_path){
    BaseFloat acoustic_scale = 1.0;
    BaseFloat lm_scale = 1.0;
    BaseFloat word_ins_penalty = 0.0;

    fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &clat);
    if (word_ins_penalty > 0.0) {
        AddWordInsPenToCompactLattice(word_ins_penalty, &clat);
    }

    // CompactLattice best_path;
    CompactLatticeShortestPath(clat, &best_path);
    
    if (best_path.Start() == fst::kNoStateId) {
    // KALDI_WARN << "Possibly empty lattice for utterance-id " << key
    //             << "(no output)";
    // n_err++;
        assert(false);
    } else {
        if (word_ins_penalty > 0.0) {
            AddWordInsPenToCompactLattice(-word_ins_penalty, &best_path);
        }
        fst::ScaleLattice(fst::LatticeScale(1.0 / lm_scale, 1.0/acoustic_scale),
                            &best_path);
    }
}



int main(int argc, char *argv[]) {

    const char *usage =
        "Creates training graphs (without transition-probabilities, by default)\n"
        "\n"
        "Usage:   compile-train-graphs [options] <tree-in> <model-in> "
        "<lexicon-fst-in> <transcriptions-rspecifier> <graphs-wspecifier>\n"
        "e.g.: \n"
        " compile-train-graphs tree 1.mdl lex.fst "
        "'ark:sym2int.pl -f 2- words.txt text|' ark:graphs.fsts\n";
    ParseOptions po(usage);


    TrainingGraphCompilerOptions gopts;
    int32 batch_size = 250;
    gopts.transition_scale = 0.0;  // Change the default to 0.0 since we will generally add the
    // transition probs in the alignment phase (since they change eacm time)
    gopts.self_loop_scale = 0.0;  // Ditto for self-loop probs.
    std::string disambig_rxfilename;
    gopts.Register(&po);
    po.Register("batch-size", &batch_size,
                "Number of FSTs to compile at a time (more -> faster but uses "
                "more memory.  E.g. 500");
    po.Register("read-disambig-syms", &disambig_rxfilename, "File containing "
                "list of disambiguation symbols in phone symbol table");



    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    LatticeFasterDecoderConfig config;
    std::string word_syms_filename;
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial, "If true, produce output even if end state was not reached.");



    bool output_if_error = true;
    bool output_if_empty = false;
    bool test = false;
    bool allow_duplicate_paths = false;
    po.Register("output-error-lats", &output_if_error, "Output lattices that aligned "
                "with errors (e.g. due to force-out");
    po.Register("output-if-empty", &output_if_empty, "If true: if algorithm gives "
                "error and produces empty output, pass the input through.");
    po.Register("test", &test, "If true, testing code will be activated "
                 "(the purpose of this is to validate the algorithm).");
    po.Register("allow-duplicate-paths", &allow_duplicate_paths, "Only "
                "has an effect if --test=true.  If true, does not die "
                "(only prints warnings) if duplicate paths are found. "
                "This should only happen with very pathological lexicons, "
                "e.g. as encountered in testing code.");
    WordAlignLatticeLexiconOpts opts;
    opts.Register(&po);


    bool print_silence = false;
    BaseFloat frame_shift = 0.01;
    BaseFloat shift_delta = 0.015;

    // int32 precision = 2;
    po.Register("print-silence", &print_silence, "If true, print optional-silence "
                "(<eps>) arcs");
    po.Register("frame-shift", &frame_shift, "Time in seconds between frames.\n");
    // po.Register("precision", &precision,
    //             "Number of decimal places for start duration times (note: we "
    //             "may use a higher value than this if it's obvious from "
    //             "--frame-shift that this value is too small");

    po.Read(argc, argv);

    std::string tree_rxfilename = po.GetArg(1);
    std::string model_rxfilename = po.GetArg(2);
    std::string lex_rxfilename = po.GetArg(3);
    std::string transcript_rspecifier = po.GetArg(4);
    // std::string fsts_wspecifier = po.GetArg(5);

    ContextDependency ctx_dep;  // the tree.
    ReadKaldiObject(tree_rxfilename, &ctx_dep);
    TransitionModel trans_model;
    ReadKaldiObject(model_rxfilename, &trans_model);
    // need VectorFst because we will change it by adding subseq symbol.
    VectorFst<StdArc> *lex_fst = fst::ReadFstKaldi(lex_rxfilename);
    std::vector<int32> disambig_syms;
    if (disambig_rxfilename != "")
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_syms))
        KALDI_ERR << "fstcomposecontext: Could not read disambiguation symbols from "
                  << disambig_rxfilename;
    TrainingGraphCompiler gc(trans_model, ctx_dep, lex_fst, disambig_syms, gopts);
    lex_fst = NULL;  // we gave ownership to gc.
    // SequentialInt32VectorReader transcript_reader(transcript_rspecifier);
    // TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);


    // std::string model_in_filename = po.GetArg(6);
    // std::string fst_in_str = po.GetArg(7);
    std::string feature_rspecifier = po.GetArg(5);
    // std::string lattice_wspecifier = po.GetArg(9);
    // std::string words_wspecifier = po.GetOptArg(10);
    // std::string alignment_wspecifier = po.GetOptArg(11);
    SequentialBaseFloatMatrixReader loglike_reader(feature_rspecifier);
    // TransitionModel trans_model;
    // ReadKaldiObject(model_in_filename, &trans_model);
    bool determinize = config.determinize_lattice;
    // CompactLatticeWriter compact_lattice_writer;
    // LatticeWriter lattice_writer;
    // if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
    //        : lattice_writer.Open(lattice_wspecifier)))
    //   KALDI_ERR << "Could not open table for writing lattices: "
    //              << lattice_wspecifier;
    // Int32VectorWriter words_writer(words_wspecifier);
    // Int32VectorWriter alignment_writer(alignment_wspecifier);
    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;



    std::string align_lexicon_rxfilename = po.GetArg(6);
    // std::string model_rxfilename = po.GetArg(2),
    // std::string lats_rspecifier = po.GetArg(3),
    // std::string lats_wspecifier = po.GetArg(4);

    std::vector<std::vector<int32>> lexicon;
    {
      bool binary_in;
      Input ki(align_lexicon_rxfilename, &binary_in);
      KALDI_ASSERT(!binary_in && "Not expecting binary file for lexicon");
      if (!ReadLexiconForWordAlign(ki.Stream(), &lexicon)) {
        KALDI_ERR << "Error reading alignment lexicon from "
                  << align_lexicon_rxfilename;
      }
    }
    // TransitionModel tmodel;
    // ReadKaldiObject(model_rxfilename, &trans_model);
    // SequentialCompactLatticeReader clat_reader(lats_rspecifier);
    // CompactLatticeWriter clat_writer(lats_wspecifier); 
    WordAlignLatticeLexiconInfo lexicon_info(lexicon);
    { std::vector<std::vector<int32> > temp; lexicon.swap(temp); }

    std::unordered_map<std::string,std::vector<int>> groudtruth_map;{
        std::cout<<"transcript_rspecifier:"<<transcript_rspecifier<<"\n";
        std::ifstream ifs(transcript_rspecifier);
        std::string line;
        while (std::getline(ifs, line)){
            if (line.empty()) continue;
            // std::cout<<"transcript:"<<line<<"\n";
            // int pos = line.find_first_of(" \t");
            // assert(pos>0);
            // std::string key(line.begin(), line.begin()+pos);
            // std::string text(line.begin()+pos+1, line.end());
            // std::stringstream ss(text);
            std::stringstream ss(line);
            std::vector<int> text_ids;
            int word_id;
            std::string key;
            ss >> key;
            while (ss >> word_id){
                text_ids.emplace_back(word_id);
            }
            groudtruth_map.emplace(key, text_ids);
        }
        ifs.close();
    }


    std::string align_output_path = po.GetArg(7);

    std::cout<<"align_output_path:"<<align_output_path<<"\n";
    std::ofstream ofs(align_output_path);


    while ( !loglike_reader.Done()){
        // assert(transcript_reader.Key() == loglike_reader.Key());
        auto start_time = std::chrono::system_clock::now();

        std::string key = loglike_reader.Key();
        VectorFst<StdArc> decode_fst;
        {
            // std::string key = transcript_reader.Key();
            // const std::vector<int32> &transcript = transcript_reader.Value();
            const std::vector<int32> &transcript = groudtruth_map[key];
            if (!gc.CompileGraphFromText(transcript, &decode_fst)) {
                decode_fst.DeleteStates();  // Just make it empty.
            }
            if (decode_fst.Start() != fst::kNoStateId) {
                // num_succeed++;
                // fst_writer.Write(key, decode_fst);
            } else {
                // KALDI_WARN << "Empty decoding graph for utterance "
                //             << key;
                assert(false);
            }
        }

        int frames;
        CompactLattice decode_clat;
        {
          Matrix<BaseFloat> loglikes (loglike_reader.Value());
          loglike_reader.FreeCurrent();
          frames = loglikes.NumRows();
          if (loglikes.NumRows() == 0) {
              assert(false);
            // KALDI_WARN << "Zero-length utterance: " << utt;
            // num_fail++;
            // continue;
          }
          DecodableMatrixScaledMapped decodable(trans_model, loglikes, acoustic_scale);
          LatticeFasterDecoder decoder(decode_fst, config);
          double like;
          std::string utt = key;
          if (LocalDecodeUtteranceLatticeFaster(
                  decoder, decodable, trans_model, word_syms, utt,
                  acoustic_scale, determinize, allow_partial, decode_clat,
                  &like)) {
          } else{
              assert(false);
          }
        }

        {
            CompactLattice clat_1;
            lat_one_best(decode_clat, clat_1);
            CompactLattice aligned_clat;
            bool ok = WordAlignLatticeLexicon(clat_1, trans_model, lexicon_info, opts,
                                        &aligned_clat);
            assert(ok);
            TopSortCompactLatticeIfNeeded(&aligned_clat);
            CompactLattice clat_2;
            lat_one_best(aligned_clat, clat_2);
            std::vector<int32> words, times, lengths;

            if (!CompactLatticeToWordAlignment(clat_2, &words, &times, &lengths)) {
                assert(false);
            } else {
                KALDI_ASSERT(words.size() == times.size() &&
                            words.size() == lengths.size());

                auto end_time = std::chrono::system_clock::now();
                float time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count()/1000.0;
                float total_time=frames*10/1000.0;
                float rtf=time_cost/total_time;
                std::string cost_info;
                cost_info = cost_info + std::to_string(time_cost)
                            +"_"+std::to_string(total_time)
                            +"_"+std::to_string(rtf);
                
                std::stringstream ss;
                ss << key;
                ss << " "<<cost_info;
                for (size_t i = 0; i < words.size(); i++) {
                    if (words[i] == 0 && !print_silence)  // Don't output anything for <eps> links, which
                        continue; // correspond to silence....
                    ss<<" "<<"("<<word_syms->Find(words[i]);
                    ss<<" "<< (i==0 ? 0 :(times[i]*frame_shift+shift_delta));
                    ss<<" "<<((times[i]+lengths[i])*frame_shift+shift_delta)<<")";
                }
                std::cout<<ss.str()<<"\n";
                ofs<<ss.str()<<"\n";
            }
        }
        // transcript_reader.Next();
        loglike_reader.Next();
    }
    ofs.close();
    delete word_syms;
}
