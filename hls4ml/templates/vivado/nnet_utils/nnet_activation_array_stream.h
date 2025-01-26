#ifndef NNET_ACTIVATION_ARRAY_STREAM_H_
#define NNET_ACTIVATION_ARRAY_STREAM_H_

#include "ap_fixed.h"
#include "hls_stream.h"
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_stream.h"
#include "nnet_types.h"
#include <cmath>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void linear(hls::stream<data_T> data[CONFIG_T::n_chan], hls::stream<res_T> res[CONFIG_T::n_chan]) {
    LinearLoop: for (int i = 0; i < CONFIG_T::n_in/CONFIG_T::n_chan; i++) {
        #pragma HLS PIPELINE

        data_T in_data[CONFIG_T::n_chan];
        #pragma HLS ARRAY_PARTITION variable=in_data complete
        for(int j = 0; j < CONFIG_T::n_chan; j++) {
            #pragma HLS UNROLL
            in_data[j] = data[j].read();
        }
        for (int j = 0; j < CONFIG_T::n_chan; j++) {
            #pragma HLS UNROLL
            res_T out_data = in_data[j];
            res[j].write(out_data);
        }
    }
}

// *************************************************
//       RELU Activation
// *************************************************
template <class data_T, class res_T, typename CONFIG_T>
void relu(hls::stream<data_T> data[CONFIG_T::n_chan], hls::stream<res_T> res[CONFIG_T::n_chan]) {

    for (int i = 0; i < CONFIG_T::n_in / CONFIG_T::n_chan; i++) { // usually this should be 1
        #pragma HLS PIPELINE

        for (int j = 0; j < CONFIG_T::n_chan; j++) {
            #pragma HLS UNROLL

            data_T in_data = data[j].read();
			res_T out_data;
			
			if (in_data > 0)
				out_data = in_data;
			else
				out_data = 0;

            res[j].write(out_data);        
        }
    }
}	

// *************************************************
//       Flatten for the MLP
// *************************************************
template<class data_T, class res_T>
void flatten_array_stream(hls::stream<data_T> data[10], hls::stream<res_T> res[500]) {
    
	data_T in_data[10];
    #pragma HLS ARRAY_PARTITION variable=in_data complete
	
	FlattenLoop: for (int i = 0; i < 500 / 10; i++) { // read streams 50 times
        #pragma HLS PIPELINE
        
        for (int j = 0; j < 10; j++) {
            #pragma HLS UNROLL
            in_data[j] = data[j].read();
        }
        
        for (int j = 0; j < 10; j++) {
            #pragma HLS UNROLL
            int out_index = i * 10 + j;
            res[out_index].write(in_data[j]);
        }
    }
}

} // namespace nnet

#endif
