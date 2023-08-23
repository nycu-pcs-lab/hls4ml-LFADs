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

} // namespace nnet

#endif
