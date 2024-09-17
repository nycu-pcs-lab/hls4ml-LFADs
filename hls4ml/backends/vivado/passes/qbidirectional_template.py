from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import Bidirectional

recr_mult_config_template = """struct config{index} : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned n_nonzeros = {nonzeros};
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef {accum_dense_t.name} accum_t;
    typedef {forward_bias_t.name} bias_t;
    typedef {forward_weight_t.name} weight_t;
    typedef ap_{index_t} index_t;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

# activation templates
hard_activ_config_template = """struct {type}_config{index} : nnet::hard_activ_config{{
    static const unsigned n_in = {n_in};
    static const {slope_t.name} slope;
    static const {shift_t.name} shift;
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
}};
const {slope_t.name} {type}_config{index}::slope = {slope};
const {shift_t.name} {type}_config{index}::shift = {shift};\n"""

recr_hard_activ_config_template = """struct {type}_config{index}_recr  : nnet::hard_activ_config {{
    static const unsigned n_in = {n_in};
    static const {slope_t.name} slope;
    static const {shift_t.name} shift;
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
}};
const {slope_t.name} {type}_config{index}_recr::slope = {slope};
const {shift_t.name} {type}_config{index}_recr::shift = {shift};\n"""

activ_config_template = """struct {type}_config{index} : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef ap_{table_t} table_t;
}};\n"""

recr_activ_config_template = """struct {type}_config{index}_recr : nnet::activ_config {{
    static const unsigned n_in = {n_in};
    static const unsigned table_size = {table_size};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
    typedef ap_{table_t} table_t;
}};\n"""

# LSTM + GRU templates

recr_forward_config_template = """struct config{index}_f : nnet::{recr_type}_config {{
    typedef {accum_dense_t.name} accum_dense_t;
    typedef {accum_t.name} accum_t;
    typedef {forward_weight_t.name} weight_t;  // Matrix
    typedef {forward_bias_t.name} bias_t;  // Vector
    typedef {config_mult_t1} mult_config1;
    typedef {config_mult_t2} mult_config2;
    typedef {recr_act_t} ACT_CONFIG_{RECR_TYPE};
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::{recurrent_activation}<x_T, y_T, config_T>;
    typedef {act_t} ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::{activation}<x_T, y_T, config_T>;
    static const unsigned n_in  = {n_in};
    static const unsigned n_out = {sub_n_out};
    static const unsigned n_state = {sub_n_state};
    static const unsigned n_sequence = {n_sequence};
    static const unsigned n_sequence_out = {n_sequence_out};
    static const unsigned io_type = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
    static const bool use_static = {static};
    typedef {state_t} state_t;
    typedef {act} act_t;
    typedef {recr_act} recr_act_t;
    static const unsigned merge_mode = nnet::{merge_mode};
}};\n"""

recr_backward_config_template = """struct config{index}_b : nnet::{recr_type}_config {{
    typedef {accum_dense_t.name} accum_dense_t;
    typedef {accum_t.name} accum_t;
    typedef {backward_weight_t.name} weight_t;  // Matrix
    typedef {backward_bias_t.name} bias_t;  // Vector
    typedef {config_mult_t1} mult_config1;
    typedef {config_mult_t2} mult_config2;
    typedef {recr_act_t} ACT_CONFIG_{RECR_TYPE};
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::{recurrent_activation}<x_T, y_T, config_T>;
    typedef {act_t} ACT_CONFIG_T;
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::{activation}<x_T, y_T, config_T>;
    static const unsigned n_in  = {n_in};
    static const unsigned n_out = {sub_n_out};
    static const unsigned n_state = {sub_n_state};
    static const unsigned n_sequence = {n_sequence};
    static const unsigned n_sequence_out = {n_sequence_out};
    static const unsigned io_type = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
    static const bool use_static = {static};
    typedef {state_t} state_t;
    typedef {act} act_t;
    typedef {recr_act} recr_act_t;
    static const unsigned merge_mode = nnet::{merge_mode};
}};\n"""

bidirectional_config_template = """struct config{index} : nnet::bidirectional_config {{
    typedef {accum_dense_t.name} accum_dense_t;
    typedef {accum_t.name} accum_t;
    typedef {backward_weight_t.name} weight_t;  // Matrix
    typedef {backward_bias_t.name} bias_t;  // Vector

    typedef {config_rnn_f} config_rnn_layer_f;
    typedef {config_rnn_b} config_rnn_layer_b;
    static const unsigned n_in  = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned n_state = {n_state};
    static const unsigned n_sequence = {n_sequence};
    static const unsigned n_sequence_out = {n_sequence_out};
    static const unsigned io_type = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const bool store_weights_in_bram = false;
    static const unsigned merge_mode = nnet::{merge_mode};
}};\n"""

recr_function_template = 'nnet::bidirectional_array<{input_t}, {output_t}, {config}>({input}, {output}, {bw}, {bwr}, {bb}, {bbr}, {fw}, {fwr}, {fb}, {fbr});'

recr_include_list = ['nnet_utils/nnet_bidirectional.h']

class BidirectionalConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__((Bidirectional))
        self.template = bidirectional_config_template
        self.act_template = activ_config_template
        self.recr_act_template = recr_activ_config_template
        self.mult1_template = recr_mult_config_template
        self.mult2_template = recr_mult_config_template
        self.rnn_forward_template = recr_forward_config_template
        self.rnn_backward_template = recr_backward_config_template

    def format(self, node):

        params = self._default_config_params(node)
        if 'hard' in node.get_attr('activation'):
            self.act_template = hard_activ_config_template
        if 'hard' in node.get_attr('recurrent_activation'):
            self.recr_act_template = recr_hard_activ_config_template
        params['n_in'] = node.get_input_variable().dim_names[1]
        params['n_sequence'] = node.get_input_variable().dim_names[0]
        if node.get_attr('return_sequences'):
            params['n_sequence_out'] = node.get_output_variable().dim_names[0]
            params['n_state'] = node.get_output_variable().dim_names[1] + '/2'
            params['n_out'] = node.get_output_variable().dim_names[1]
        else:
            params['n_sequence_out'] = 1
            params['n_state'] = node.get_output_variable().dim_names[0] + '/2'
            params['n_out'] = node.get_output_variable().dim_names[0]
        params['config_mult_t1'] = f'config{node.index}_1'
        params['config_mult_t2'] = f'config{node.index}_2'
        params['config_rnn_f'] = f'config{node.index}_f'
        params['config_rnn_b'] = f'config{node.index}_b'
        params['recr_act_t'] = '{}_config{}_recr'.format(node.get_attr('recurrent_activation'), node.index)
        params['act_t'] = '{}_config{}'.format(node.get_attr('activation'), node.index)
        params['strategy'] = node.get_attr('strategy')
        params['static'] = 'true' if node.attributes['static'] else 'false'
        params['recr_type'] = node.get_attr('subclass_name').lower()[1:]
        params['RECR_TYPE'] = node.get_attr('subclass_name')[1:]
        params['state_t'] = 'state{}_t'.format(node.index)

        if 'hard' in node.get_attr('recurrent_activation'):
            params['recr_act'] = 'recr_act{}_t'.format(node.index)
        if 'hard' in node.get_attr('activation'):
            params['act'] = 'act{}_t'.format(node.index)

        if node.class_name == 'LSTM':
            n_recr_mult = 4
        else:  # GRU
            n_recr_mult = 3

        bidirectional_config = self.template.format(**params)

        recr_forward_config = self.rnn_forward_template.format(**params)
        recr_backward_config = self.rnn_backward_template.format(**params)

        act_params = self._default_config_params(node)
        recr_act_params = self._default_config_params(node)

        act_params['type'] = node.get_attr('activation')
        recr_act_params['type'] = node.get_attr('recurrent_activation')
        if node.get_attr('return_sequences'):
            act_params['n_in'] = node.get_output_variable().dim_names[1] + '/2'
            recr_act_params['n_in'] = node.get_output_variable().dim_names[1] + ' * %i /2' % (n_recr_mult - 1)
        else:
            act_params['n_in'] = node.get_output_variable().dim_names[0] + '/2'
            recr_act_params['n_in'] = node.get_output_variable().dim_names[0] + ' * %i /2' % (n_recr_mult - 1)

        act_config = self.act_template.format(**act_params)
        recr_act_config = self.recr_act_template.format(**recr_act_params)

        mult_params1 = self._default_config_params(node)
        mult_params2 = self._default_config_params(node)

        mult_params1['n_in'] = node.get_input_variable().dim_names[1]
        if node.get_attr('return_sequences'):
            mult_params1['n_out'] = node.get_output_variable().dim_names[1] + ' * %i /2' % n_recr_mult
        else:
            mult_params1['n_out'] = node.get_output_variable().dim_names[0] + ' * %i /2' % n_recr_mult
        mult_params1['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('forward_weight').type.precision
        )
        # 240917 crchen set reuse as nin
        #mult_params1['reuse'] = node.attributes['recurrent_reuse_factor']
        mult_params1['reuse'] = mult_params1['n_in']
        
        mult_params1['index'] = str(node.index) + '_1'
        mult_params1['nzeros'] = node.get_weights('forward_weight').nzeros
        mult_params1['nonzeros'] = node.get_weights('forward_weight').nonzeros
        if node.get_attr('return_sequences'):
            mult_params2['n_in'] = node.get_output_variable().dim_names[1] + '/2'
            mult_params2['n_out'] = node.get_output_variable().dim_names[1] + ' * %i /2' % n_recr_mult
        else:
            mult_params2['n_in'] = node.get_output_variable().dim_names[0] + '/2'
            mult_params2['n_out'] = node.get_output_variable().dim_names[0] + ' * %i /2' % n_recr_mult
        mult_params2['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_weights('forward_recurrent_weight').type.precision
        )
        # 240917 crchen set reuse as nin
        #mult_params2['reuse'] = node.attributes['recurrent_reuse_factor']
        mult_params2['reuse'] = mult_params2['n_in']
        
        mult_params2['index'] = str(node.index) + '_2'
        mult_params2['nzeros'] = node.get_weights('forward_recurrent_weight').nzeros
        mult_params2['nonzeros'] = node.get_weights('forward_recurrent_weight').nonzeros

        mult_config1 = self.mult1_template.format(**mult_params1)
        mult_config2 = self.mult2_template.format(**mult_params2)

        return mult_config1 + '\n' + mult_config2 + '\n' + recr_act_config + '\n' + act_config + '\n' + recr_forward_config + '\n' + recr_backward_config + '\n' + bidirectional_config


class BidirectionalFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__((Bidirectional), include_header=recr_include_list)
        self.template = recr_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['fw'] = node.get_weights('forward_weight').name
        params['fb'] = node.get_weights('forward_bias').name
        params['fwr'] = node.get_weights('forward_recurrent_weight').name
        params['fbr'] = node.get_weights('forward_recurrent_bias').name
        params['bw'] = node.get_weights('backward_weight').name
        params['bb'] = node.get_weights('backward_bias').name
        params['bwr'] = node.get_weights('backward_recurrent_weight').name
        params['bbr'] = node.get_weights('backward_recurrent_bias').name
        #params['activation'] = node.get_attr('activation')
        #params['recurrent_activation'] = node.get_attr('recurrent_activation')
        #params['recr_type'] = node.class_name.lower()

        return self.template.format(**params)