#ifndef NNET_DENSE_ARRAY_STREAM_H_
#define NNET_DENSE_ARRAY_STREAM_H_

#include "nnet_common.h"
#include "nnet_types.h"
#include "hls_stream.h"
#include <math.h>
#include <assert.h>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void dense_wrapper(
    data_T data[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out]
) {
    #pragma HLS INLINE recursive 
    if (CONFIG_T::strategy == nnet::latency) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        dense_latency<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_resource<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void dense(
    hls::stream<data_T> data_stream[CONFIG_T::n_in],
    hls::stream<res_T>  res_stream[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_out])
{
    data_T data[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=data complete

    res_T res[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=res complete

    Data: for (int i = 0; i< CONFIG_T::n_in; i++) {
        #pragma HLS UNROLL
        data_T data_pack = data_stream[i].read();
        data[i] = data_pack;
    }

    dense_wrapper<data_T, res_T, CONFIG_T>(data, res, weights, biases);

    Res: for (int i = 0; i < CONFIG_T::n_out; i++) {
        #pragma HLS UNROLL
        res_T res_pack = res[i];
        res_stream[i].write(res_pack);
    }
}


}

#endif