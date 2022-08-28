// bin/post-to-smat.cc

// Copyright 2017   Johns Hopkins University (Author: Daniel Povey)

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


#include "hmm/transition-model.h"
#include "util/kaldi-io.h"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace kaldi;

int main(int argc, char *argv[]){

    TransitionModel *trans_model= new TransitionModel();
    if (argc <= 0){
        std::cout<<"usage example: extract_transition2pdf_vector final_model.path phoneid output_path"<<std::endl;
    }
    std::string model_path = std::string(argv[1]);
    std::string phone_str = std::string(argv[2]);

    int phone_id;
    std::stringstream ss(phone_str); ss>>phone_id;

    try {
      bool binary;
	    Input ki(model_path, &binary);
	    trans_model->Read(ki.Stream(), binary);

    } catch (std::runtime_error& e) {
      std::cout << "Error loading the acoustic model: "<<std::endl;
    }

    for (int i=0; i<trans_model->NumTransitionIds(); i++){
      if (trans_model->TransitionIdToPhone(i+1) == phone_id){ //TransitionIdToPhone start from 1
        std::cout<<trans_model->TransitionIdToPdf(i+1)<<"\n";
      }
    }   
}
