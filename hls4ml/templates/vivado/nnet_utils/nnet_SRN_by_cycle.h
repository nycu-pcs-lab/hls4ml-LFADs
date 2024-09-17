
#ifndef NNET_SRN_STREAM_H_
#define NNET_SRN_STREAM_H_

#include "hls_stream.h"
#include "hls_math.h"
#include "nnet_common.h"

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

    data_T gen_next_random() {
        ap_int<32> rnd[N];
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
           rnd[i]  = grng[i].gen_next_random();
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




// array stream
template <class data_T, class res_T, typename CONFIG_T>
void srn(hls::stream<data_T> data_stream[CONFIG_T::n_elem], 
          hls::stream<res_T> res_stream[CONFIG_T::n_elem],  
          GRNGArray<CONFIG_T::n_elem, CONFIG_T::n_samples,  typename CONFIG_T::seed_t> &normal
) {
        /*
        data_T inp[CONFIG_T::n_elem];
        for (unsigned j = 0; j < CONFIG_T::n_elem; j++) {
            #pragma HLS UNROLL
            inp[j] = data_stream[j].read();
		}
    // use the mean of  input as seed
    // but require to shift so it is an integer
    
   data_T sum=0;
   for (unsigned j = 0; j < CONFIG_T::n_elem; j++) {
            #pragma HLS PIPELINE
            sum += inp[j];
		}
    data_T mean = sum / CONFIG_T::n_elem;
    std::cout<<"mean: "<< mean <<std::endl;
    std::cout<<"mean binary: "<< mean.to_string(2) <<std::endl; 
    ap_uint<32>  mean_int = mean * (1 << (mean.width - mean.iwidth));
    std::cout<<"mean_int binary: "<< mean_int.to_string(2) <<std::endl;
    std::cout<<"mean_int, use as seed: "<< mean_int <<std::endl;
    
 GRNGArray<CONFIG_T::n_elem, CONFIG_T::n_samples, res_T> normal(mean_int);
 */
 std::cout<<"USE GRNG CYCLE BY CYCLE" <<std::endl;
   data_T inp[CONFIG_T::n_elem];
      for (unsigned j = 0; j < CONFIG_T::n_elem; j++) {
          #pragma HLS UNROLL
          inp[j] = data_stream[j].read();
		}     
   res_T rnd[CONFIG_T::n_elem];


   //normal_mean.next(rnd);
   normal.next(rnd);
   

		RNGLoop: for (int j = 0; j < CONFIG_T::n_elem; j++) {
			#pragma HLS UNROLL
			//res_pack[j] = rnd[j] + inp[j];
			//res_pack[j] = rnd[j];
            res_stream[j].write(rnd[j]);
		}

        

 
 }
    
    
}


#endif