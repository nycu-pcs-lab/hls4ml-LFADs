
#ifndef NNET_EXPLOGVAR_STREAM_H_
#define NNET_EXPLOGVAR_STREAM_H_

#include "ap_fixed.h"
#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_stream.h"
#include "nnet_types.h"
#include <cmath>
#include <bitset>
namespace nnet {

// *************************************************
//       Exp logvar => exp(0.5 * inputs) 
// *************************************************

inline float explogvar_fcn_float(float input) { return std::exp(0.5 * input); }

template <class data_T, typename CONFIG_T> inline float explogvar_real_val_from_idx(unsigned i) {
    // Treat the index as the top N bits
    static constexpr int N = ceillog2(CONFIG_T::table_size); // number of address bits for table
    data_T x(0);
	//std::cout<<"i: "<<std::bitset<16>(i)<<std::endl; 
    x(x.width - 1, x.width - N) = i;
	//std::cout<<"x.width - 1: "<< x.width - 1 <<std::endl;
	//std::cout<<"x.width - N: "<< x.width - N <<std::endl;
	//std::cout<<"x(x.width - 1, x.width - N): "<< x.to_string(2) <<std::endl; // 2 for binary
    return (float)x;
}

template <class data_T, typename CONFIG_T> inline unsigned explogvar_idx_from_real_val(data_T x) {
    // Slice the top N bits to get an index into the table
    static constexpr int N = ceillog2(CONFIG_T::table_size); // number of address bits for table
    ap_uint<N> y = x(x.width - 1, x.width - N);              // slice the top N bits of input
    return (unsigned)y(N - 1, 0);
}

template <class data_T, typename CONFIG_T>
void init_explogvar_table(typename CONFIG_T::exp_table_t table_out[CONFIG_T::table_size]) {
    // The template data_T is the data type used to address the table
    for (unsigned i = 0; i < CONFIG_T::table_size; i++) {
        // Slicing bits for address is going to round towards 0, so take the central value
        float x = explogvar_real_val_from_idx<data_T, CONFIG_T>(i);
        typename CONFIG_T::exp_table_t exp_x = explogvar_fcn_float(x);
        table_out[i] = exp_x;
    }
}

struct explogvar_config {
    static const unsigned n_elem = 32;
    static const unsigned table_size = 1024;
    typedef ap_fixed<18, 8> exp_table_t;
};

template <class data_T, class res_T, typename CONFIG_T>
void explogvar(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_explogvar_table<typename data_T::value_type, CONFIG_T>(exp_table);
        initialized = true;
    }


	
	
ExpLogVarExpLoop:
    for (unsigned i = 0; i < CONFIG_T::n_elem / data_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_pack = data.read();
		
		res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)
		
		ExpLogVarExpPackLoop:
        for (unsigned j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            unsigned x = explogvar_idx_from_real_val<typename data_T::value_type, CONFIG_T>(in_pack[j]);
            out_pack[j] = exp_table[x];
        }
       
        res.write(out_pack);
    }
}


template <class data_T, class res_T, typename CONFIG_T>
void explogvar(hls::stream<data_T> data[CONFIG_T::n_elem], hls::stream<res_T> res[CONFIG_T::n_elem]) {
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_explogvar_table<data_T, CONFIG_T>(exp_table);
        initialized = true;
    }

        data_T in_pack[CONFIG_T::n_elem];
        #pragma HLS ARRAY_RESHAPE variable=in_pack complete
        ExpLogVarExpLoop:
        for (unsigned j = 0; j < CONFIG_T::n_elem; j++) {
            #pragma HLS UNROLL
            in_pack[j] = data[j].read();
            
		}
		res_T out_pack[CONFIG_T::n_elem];
        #pragma HLS ARRAY_RESHAPE variable=out_pack complete
		
		ExpLogVarExpPackLoop:
        for (unsigned j = 0; j < CONFIG_T::n_elem; j++) {
            #pragma HLS UNROLL
            unsigned x = explogvar_idx_from_real_val<data_T, CONFIG_T>(in_pack[j]);
            out_pack[j] = exp_table[x];
        }
        for (unsigned j = 0; j < CONFIG_T::n_elem; j++) {
            #pragma HLS UNROLL
            res[j].write(out_pack[j]);
		}
}

} // namespace nnet

#endif