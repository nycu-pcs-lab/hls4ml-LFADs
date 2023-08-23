#ifndef NNET_SEPARABLE_CONV1D_STREAM_H_
#define NNET_SEPARABLE_CONV1D_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_conv1d_stream.h"
#include "nnet_sepconv_stream.h"

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void pointwise_mult_buffer_array(
    const data_T data[CONFIG_T::n_chan],
    hls::stream<res_T> res_stream[CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) {
    #pragma HLS INLINE

    res_T res[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=res complete

    data_T data_save[CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=data_save complete
    InitData: for (int id = 0; id < CONFIG_T::n_chan; id++) {
        #pragma HLS UNROLL
        data_save[id] = data[id];
    }

    #pragma HLS INLINE recursive 
    if (CONFIG_T::strategy == nnet::latency) {
        dense_latency<data_T, res_T, typename CONFIG_T::mult_config>(data_save, res, weights, biases);
    } else {
        dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(data_save, res, weights, biases);
    }

    Write: for (unsigned i = 0; i < CONFIG_T::n_filt; i++) {
        #pragma HLS UNROLL
        res_stream[i].write(res[i]);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_cl(
    hls::stream<data_T> data[CONFIG_T::n_chan],
    hls::stream<res_T>  res[CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::filt_width == 1);

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=biases complete

    data_T data_store[CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=data_store complete
    
    ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
        if (CONFIG_T::strategy == nnet::latency) {
            #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        }
        if (i_iw % CONFIG_T::stride_width == 0) {
            
            for(int j = 0; j < CONFIG_T::n_chan; j++){
                #pragma HLS UNROLL
                data_store[j] = data[j].read();
            }
            pointwise_mult_buffer_array<data_T, res_T, CONFIG_T>(data_store, res, weights, biases);
        } else {
            for(int j = 0; j < CONFIG_T::n_chan; j++){
                #pragma HLS UNROLL
                data_store[j] = data[j].read();
            }
        }
    }
}

} // namespace nnet
#endif
