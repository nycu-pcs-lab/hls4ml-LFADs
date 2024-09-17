#ifndef NNET_SRN_STREAM_H_
#define NNET_SRN_STREAM_H_

#include "hls_stream.h"
#include "hls_math.h"
#include "nnet_common.h"
#include "ap_fixed.h"
#include "nnet_stream.h"
#include "nnet_types.h"
#include <cmath>
#include <bitset>

namespace nnet {



/* The base implementation, from Wikipedia */
ap_uint<32> xorshift32() {
    /* The state must be initialized to non-zero */
    static ap_uint<32> state = 42;

    ap_uint<32> x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    state = x;
    return x;
}

/* This one is for generation of a single uniform random number (uint<32>) */
class RNG {
  public:
    RNG(ap_uint<32> seed) : state(seed) {}

    ap_uint<32> next() {
        // Due to the read and update of state, this cannot be called in parallel
        ap_uint<32> x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;

        state = x;
        return x;
    }

  private:
    ap_uint<32> state;
};

/* Generates an array of uniformly-distributed random numbers in 1 clock cycle. */
template <unsigned N> class RNGArray {
  public:
    RNGArray() {
        #pragma HLS ARRAY_PARTITION variable=state complete
    }

    RNGArray(ap_uint<32> seed) {
        #pragma HLS ARRAY_PARTITION variable=state complete
        set_seed(seed);
    }

    void next(ap_int<32> rnd[N]) {
        //#pragma HLS INLINE
        RandomGenLoop: for (int i = 0; i < N; i++) {
            #pragma HLS UNROLL

            ap_uint<32> x = state[i];
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;

            state[i] = x;
            rnd[i] = x;
        }
    }
    
    ap_uint<32> set_seed(ap_uint<32> initial_state) {
        ap_uint<32> current_state = initial_state;
        SeedGenLoop: for (int i = 0; i < N; i++) {
            // We use SplitMix32, a different family of RNG to generate the seeds,
            // to avoid common pitfalls with seed selection
            // Implementation from Kaito Udagawa, licensed under CC0
            ap_uint<32> z = (current_state += 0x9e3779b9);
            z = (z ^ (z >> 16)) * 0x85ebca6b;
            z = (z ^ (z >> 13)) * 0xc2b2ae35;
            state[i] = z ^ (z >> 16);
            current_state = state[i];
        }
        return current_state;
    }

  private:
    ap_uint<32> state[N];
};

template <unsigned N, class data_T> class GRNG {
  public:
    GRNG() {
        scale = 1. / std::sqrt(N * 1537723804776605696.);
    }

    GRNG(ap_uint<32> seed) : rng_array(seed) {
        scale = 1. / std::sqrt(N * 1537723804776605696.);
    }

    data_T next() {
        ap_int<32> rnd[N];
         #pragma HLS ARRAY_PARTITION variable=rnd complete
        rng_array.next(rnd);

        ap_int<32 + N> rnd_normal = 0;
        SampleSumLoop: for (int i = 0; i < N; i++) {
            #pragma HLS UNROLL
            rnd_normal += rnd[i];
        }

        // The implementation of multiplication can be controlled, DSP or fabric
        // #pragma HLS BIND_OP op=mul impl=fabric // Vitis, comment out for DSP
        // On Vivado, this is done with RESOURCE pragma, but doesn't work here.
        return rnd_normal * scale;
    }
    
    ap_uint<32> set_seed(ap_uint<32> initial_state) {
        ap_uint<32> new_state = rng_array.set_seed(initial_state);
        return new_state;
    }

  private:
    RNGArray<N> rng_array;
    ap_fixed<64, 0> scale;
};

template <unsigned SIZE, unsigned N, class data_T> class GRNGArray {
  public:
    GRNGArray(ap_uint<32> seed) {
        #pragma HLS ARRAY_PARTITION variable=grng complete
        ap_uint<32> current_state = seed;
        NextRandArrayLoop: for (int i = 0; i < SIZE; i++) {
            current_state = grng[i].set_seed(current_state);
        }
    }

    void next(data_T rnd[SIZE]) {
        NextRandArrayLoop: for (int i = 0; i < SIZE; i++) {
            #pragma HLS UNROLL
           rnd[i]  = grng[i].next();
        }
    }

  private:
    GRNG<N, data_T> grng[SIZE];
};

// For generating single samples
//GRNG<N_SAMPLES, result_t> normal(SEED);

// For generating array of samples



struct srn_config {
    static const unsigned n_elem = 32;
	  static const unsigned seed = 42;
    static const unsigned n_samples = 4;
    typedef ap_fixed<8, 3> seed_t;
   
};

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
void explogvar(hls::stream<data_T> data[CONFIG_T::n_elem], hls::stream<res_T> res[CONFIG_T::n_elem]) {
 

        data_T in_pack[CONFIG_T::n_elem];
        
        ExpLogVarExpLoop:
        for (unsigned j = 0; j < CONFIG_T::n_elem; j++) {
            #pragma HLS UNROLL
            
            in_pack[j] = data[j].read();
            
		}
		res_T out_pack[CONFIG_T::n_elem];
        PRAGMA_DATA_PACK(out_pack)
		
		ExpLogVarExpPackLoop:
        for (unsigned j = 0; j < CONFIG_T::n_elem; j++) {
            #pragma HLS UNROLL
            
        }
        for (unsigned j = 0; j < CONFIG_T::n_elem; j++) {
            #pragma HLS UNROLL
            res[j].write(out_pack[j]);
		}
}


struct config17 : explogvar_config {
    static const unsigned n_elem = 64;
    static const unsigned table_size = 1024;
    typedef  ap_fixed<18,8,AP_RND_CONV,AP_SAT>  exp_table_t;
};



// array stream
template <class mean_T, class explogvar_T, class res_T>
void sample(
          hls::stream<mean_T> mean_stream[64], 
          hls::stream<explogvar_T> explogvar_stream[64],
          hls::stream<res_T> res_stream[64],  
          GRNGArray<64, 4, ap_fixed<8,3> > &normal
) {
   // Initialize the lookup tables
   
  #ifdef __HLS_SYN__
      bool initialized = false;
      ap_fixed<18,8,AP_RND_CONV,AP_SAT> exp_table[1024];
  #else
      static bool initialized = false;
      static ap_fixed<18,8,AP_RND_CONV,AP_SAT> exp_table[1024];
  
  #endif
      if (!initialized) {
          // Note we are exponentiating the inputs, which have type data_T
          init_explogvar_table<explogvar_T, config17>(exp_table);
          initialized = true;
      }

    
 std::cout<<"USE EXPLOGVAR GAUSSIAN SAMPLE"<< std::endl;
    
   mean_T mean[64];
   #pragma HLS ARRAY_PARTITION variable=mean complete
   
   explogvar_T explogvar[64];
   #pragma HLS ARRAY_PARTITION variable=explogvar complete
   
   ap_fixed<8,3> rnd[64];
   #pragma HLS ARRAY_PARTITION variable=rnd complete
   

    for (unsigned j = 0; j < 64; j++) {
        #pragma HLS UNROLL
        mean[j] = mean_stream[j].read();
        explogvar[j] = exp_table[explogvar_idx_from_real_val<explogvar_T, config17>(explogvar_stream[j].read())];
		}  
  
  //#pragma HLS pipeline
  normal.next(rnd);
   
		for (int j = 0; j < 64; j++) {
			#pragma HLS UNROLL
      res_T res_data =  mean[j] + rnd[j]*explogvar[j];
      res_stream[j].write(res_data);
		}

        

 
}
    
    
}


#endif