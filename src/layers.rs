//TODO: f32 and i32 (native tensorflow index) used here as well as u64 for usize. need to either specify precision globally in lib or generically.
//a layer defines how each layer in the network communicates with the next layer.
//
//this includes but is not limited to: convolution, residual connections (how the network layers are connected),
//bias operations, recurrent connections, variant and invariant scaling, etc. if it isnt an activation function
//it should be defined here.
//there is only the sequential layer builder, any parallel stuff must be built using concat and
//split type operations or noop type layers where towers may have different length.
//this is for simplicity of the framework and may be extended on in the future.
use half::f16;
use tensorflow::ops;
use tensorflow::ops::TensorArrayV3;
use tensorflow::train::AdadeltaOptimizer;
use tensorflow::train::GradientDescentOptimizer;
use tensorflow::train::MinimizeOptions;
use tensorflow::train::Optimizer;
use tensorflow::BFloat16;
use tensorflow::Code;
use tensorflow::DataType;
use tensorflow::Graph;
use tensorflow::Operation;
use tensorflow::Output;
use tensorflow::OutputName;
use tensorflow::SavedModelBundle;
use tensorflow::SavedModelSaver;
use tensorflow::Scope;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Shape;
use tensorflow::SignatureDef;
use tensorflow::Status;
use tensorflow::Tensor;
use tensorflow::TensorInfo;
use tensorflow::TensorType;
use tensorflow::Variable;
use tensorflow::REGRESS_INPUTS;
use tensorflow::REGRESS_METHOD_NAME;
use tensorflow::REGRESS_OUTPUTS;

use crate::activations::*;

//NOTE: almost every method is a trait since this framework aims to be as
//      extensible as possible

//TODO: use a trait for layer type such as in Architecture.rs. this will organize
//      layers like conv, merge and split that are different into the same object. also layers
//      defined in native tf as ops can be easily integrated. use into(tf::op) blanket
//      implementation as the result of Layer and lazy builder trait to support this (e.g.:  layer
//      without size known yet still will impl into)
//TODO: traits need refactoring, they are Tensorflow specific for now but this framework aims to
//      abstract beyond that. Still is future proof for all possible tensorflow ops.

//TODO: Tensorflow operations should be copy they are just pointers we cant do anything about the underlying data clone also should get optimized anyways so its just boilerplate. If anything put a guard on it and make it send/sync as well.

//LAYER TRAITS//
///Builds a tensorflow layer from an arbitrary type and a graph scope.
///Anything that successfully implements this can be used as a layer in a Brain.
pub trait BuildLayer {
    ///Build and insert the layer configuration into the tensorflow graph
    fn build_layer(&self, scope: &mut Scope) -> Result<TrainnableLayer, Status>;
}
pub trait AccessLayer {
    fn get_input(&self) -> &Option<Operation>;
    fn get_input_size(&self) -> u64;
    fn get_output_size(&self) -> u64;
    fn get_width(&self) -> u64;
    fn get_activation(&self) -> &Option<Activation>;
    fn get_dtype(&self) -> DataType;
}
///A trait that defines the standard layer parameters via getters
///and setters that can mutably configure layers, used by internal network builder.
pub trait ConfigureLayer {
    fn input(self, input: Operation) -> Self;
    fn input_size(self, input_size: u64) -> Self;
    fn output_size(self, output_size: u64) -> Self;
    fn width(self, width: u64) -> Self;
    fn activation(self, activation: Option<Activation>) -> Self;
    fn dtype(self, dtype: DataType) -> Self;
}

///Layer constructor trait with default values.
pub trait InitializeLayer {
    fn init() -> Self
    where
        Self: Sized;
}
///Set all user defined parameters for a layer.
pub trait build_layer {
    fn new(width: u64, activation: Activation) -> Self
    where
        Self: Sized;
}
impl<T> build_layer for T
where
    T: InitializeLayer + ConfigureLayer,
{
    fn new(width: u64, activation: Activation) -> Self {
        Self::init().width(width).activation(Some(activation))
    }
}

//FUNDAMENTAL LAYER STATE//
///A subset of concrete state standardized to define any layer in tf.
pub struct LayerState {
    input: Option<Operation>, //this trait bound is named O1 in tensorflow-rust
    //input output vector sizes
    input_size: u64,
    output_size: u64,
    //external configuration for vector sizes
    width: u64,
    //TODO: remove dyn here
    activation: Option<Activation>,
    dtype: DataType,
}
impl ConfigureLayer for LayerState {
    fn input(mut self, input: Operation) -> Self {
        self.input = Some(input);
        self
    }
    fn input_size(mut self, input_size: u64) -> Self {
        self.input_size = input_size;
        self
    }
    fn output_size(mut self, output_size: u64) -> Self {
        self.output_size = output_size;
        self
    }
    fn width(mut self, width: u64) -> Self {
        self.width = width;
        self
    }
    fn activation(mut self, activation: Option<Activation>) -> Self {
        self.activation = activation;
        self
    }
    fn dtype(mut self, dtype: DataType) -> Self {
        self.dtype = dtype;
        self
    }
}
impl AccessLayer for LayerState {
    fn get_input(&self) -> &Option<Operation> {
        &self.input
    }
    fn get_input_size(&self) -> u64 {
        self.input_size
    }
    fn get_output_size(&self) -> u64 {
        self.output_size
    }
    fn get_width(&self) -> u64 {
        self.width
    }
    fn get_activation(&self) -> &Option<Activation> {
        &self.activation
    }
    fn get_dtype(&self) -> DataType {
        self.dtype
    }
}
impl InitializeLayer for LayerState {
    fn init() -> Self {
        LayerState {
            input: None,
            input_size: 0,
            output_size: 0,
            width: 0,
            activation: None,
            dtype: DataType::Float,
        }
    }
}
///A layer that has been added to a graph with its Output and backing
///Operation ready to be assigned and trainnable variables exposed.
pub struct TrainnableLayer {
    pub variables: Vec<Variable>,
    pub output: Output,
    pub operation: Operation,
}

pub trait BuildNetwork {
    fn build(self, Scope: &mut Scope) -> Result<Vec<TrainnableLayer>, Status>;
}
//TODO: this should be a derive macro for any T | T.0 = LayerState
pub trait InheritState {
    fn get_mut_layer_state(&mut self) -> &mut LayerState;
    fn get_layer_state(&self) -> &LayerState;
}
///blanket for optional inherited layer state
impl<L> ConfigureLayer for L
where
    L: InheritState,
{
    fn input(mut self, input: Operation) -> Self {
        self.get_mut_layer_state().input = Some(input);
        self
    }
    fn input_size(mut self, input_size: u64) -> Self {
        self.get_mut_layer_state().input_size = input_size;
        self
    }
    fn output_size(mut self, output_size: u64) -> Self {
        self.get_mut_layer_state().output_size = output_size;
        self
    }
    fn width(mut self, width: u64) -> Self {
        self.get_mut_layer_state().width = width;
        self
    }
    fn activation(mut self, activation: Option<Activation>) -> Self {
        self.get_mut_layer_state().activation = activation;
        self
    }
    fn dtype(mut self, dtype: DataType) -> Self {
        self.get_mut_layer_state().dtype = dtype;
        self
    }
}
impl<L> AccessLayer for L
where
    L: InheritState,
{
    fn get_input(&self) -> &Option<Operation> {
        &self.get_layer_state().input
    }
    fn get_input_size(&self) -> u64 {
        self.get_layer_state().input_size
    }
    fn get_output_size(&self) -> u64 {
        self.get_layer_state().output_size
    }
    fn get_width(&self) -> u64 {
        self.get_layer_state().width
    }
    fn get_activation(&self) -> &Option<Activation> {
        &self.get_layer_state().activation
    }
    fn get_dtype(&self) -> DataType {
        self.get_layer_state().dtype
    }
}

//LAYER DEFINITIONS//
//each layer must derive InheritState accessor trait,
//BuildLayer and InitializeLayer
//
//BuildLayer is the trait that defines the layers architecture, the rest are setters, getters and
//defaults.
pub struct std_layer(LayerState);
impl InheritState for std_layer {
    fn get_mut_layer_state(&mut self) -> &mut LayerState {
        &mut self.0
    }
    fn get_layer_state(&self) -> &LayerState {
        &self.0
    }
}
impl BuildLayer for std_layer {
    fn build_layer(&self, scope: &mut Scope) -> Result<TrainnableLayer, Status> {
        //instead use the accessor
        let input_size = self.get_input_size();
        //let output_size = self.0.output_size;
        let output_size = self.get_output_size();
        //tensorflow status
        //let input = self.0.input.clone().unwrap(); //TODO: propagate this with the weird status
        let input = self.get_input().clone().unwrap();
        //let dtype = self.0.dtype;
        let dtype = self.get_dtype();
        let mut scope = scope.new_sub_scope("layer"); //TODO: layer counter or some uuid
        let scope = &mut scope;
        let w_shape = ops::constant(&[input_size as i64, output_size as i64][..], scope)?;
        let mut init_bias: Tensor<f32> = Tensor::new(&[1u64, output_size as u64]);
        for i in 0..output_size {
            init_bias[i as usize] = 0.0 as f32;
        }

        let w = Variable::builder()
            .initial_value(
                ops::RandomStandardNormal::new()
                    //.dtype(DataType::Float)
                    .dtype(dtype)
                    .build(w_shape, scope)?,
            )
            //.data_type(DataType::Float)
            .data_type(dtype)
            .shape([input_size, output_size])
            .build(&mut scope.with_op_name("w"))?;

        let b = Variable::builder()
            .initial_value(
                //this is more sensible than random by the nature of the bias,
                //otherwise we will miss correlated signals by default
                //with RandomStandardNormal, albeit with a little longer
                //trainning time.
                ops::Cast::new().SrcT(DataType::Float).DstT(dtype).build(
                    ops::constant(
                        Tensor::new(&[1u64, output_size as u64])
                            .with_values(&vec![0.0f32; output_size as usize][..])?,
                        scope,
                    )?,
                    scope,
                )?, // ops::RandomStandardNormal::new()
                    //     .dtype(DataType::Float)
                    //     .build(ops::constant(&[1i64,output_size as i64][..], scope)?, scope)?,
            )
            //.data_type(DataType::Float)
            .data_type(dtype)
            .shape(&[1i64, output_size as i64][..])
            .build(&mut scope.with_op_name("b"))?;

        //n is input_size to be divided at each node in order to normalize the signals at each node before activation
        let act = (self.get_activation().as_ref().unwrap())(
            ops::add(
                ops::mat_mul(input.clone(), w.output().clone(), scope)?,
                b.output().clone(),
                scope,
            )?
            .into(),
            scope,
        )?;

        let output_op = act.clone();
        Ok(TrainnableLayer {
            variables: vec![w.clone(), b.clone()],
            output: act.into(),
            operation: output_op,
        })
    }
}
//TODO: this should be solved also with the derive macro for InheritState
impl InitializeLayer for std_layer {
    fn init() -> Self {
        std_layer(LayerState {
            input: None,
            input_size: 0,
            output_size: 0,
            width: 0,
            activation: None,
            dtype: DataType::Float,
        })
    }
}
//TODO: fix lazy builder
//pub fn std() -> Vec<Box<std_layer>> {
//    vec![Box::new(std_layer::init())]
//}
///================
///----NORM_NET----
///================
/// A standard fully connected layer without bias trainnable parameters
/// instead normalizing and dropping out connections at each node.
///
/// NOTE: currently inputs and outputs must be flattened if representing >1 dim data
///
/// #
/// PROS:
///
/// * better exploration by removing instabilities inherent to bias
///
/// * gradient based connection-wise dropout with tan weights
///
/// * better transfer learning by removing bias connections
///
/// * ~shouldn't~ have exploding gradient although vanishing gradient may be possible due to normalizing division.
///
/// * technically connection wise dropout is divisive (top down) architecture search (e.g.: the opposite of NEAT (bottom up) which is agglomerative)
///
/// CONS:
/// * may be slower due to more operations (division)
///
/// * input and output must/should be tailored for normalized input/output (standard data science practices)
///
/// * needs large type precision for stability, but the stability can be tuned (as apposed to bias which needs architectural considerations)
///
/// #
/// NOTE: parameters goes to zero whereas biases find some 1-dimensional partition from -inf to inf. This helps
/// build subgraph search modules (subtrees essentially). That can quickly optimize for distinct domains and labels via dropout.
///
/// NOTE:
/// Tanh should the first and last layers activation function to map inputs and outputs to negative values.
/// multiplying the output by some multiple of 10 allows the otherwise normalized network to take in and output whole integers.
/// -x > y > x | x > 1
///
/// NOTE:
/// We dont use BFloat since the integer range is only used as a buffer for addition overflow in matmul.
/// In all other operations we are strictly bounded -1 > x > 1. As long as layer_width is not
/// greater than Float range we are fine in the worst case (summing all 1's).
/// Otherwise decimal precision of float type is our parameter type precision.
pub struct norm_layer {
    //TODO: accessor methods for this so std_layer.* is how configuration is set and ultimately evaluated
    conf: LayerState,
    order: u64, //the effective precision of the network
    weights: Option<Variable>,
    //TODO: this should be encapsulated in the currently deprecated mod
}
////TODO: this should be for self==GraphNode so we can method chain or return a trait bound
//impl BuildLayer for norm_layer {
//    fn build_layer(&self, scope: &mut Scope) -> Result<TrainnableLayer, Status> {
//        //fn norm_layer<GraphNode: Into<Output>>(
//        //   input: GraphNode,
//        //   input_size: u64,
//        //   output_size: u64,
//        //   activation: &dyn Fn(Output, &mut Scope) -> Result<Operation, Status>,
//        //   scope: &mut Scope,
//        //) -> Result<(Vec<Variable>, Output, Operation), Status> {
//        let input_size = self.conf.input_size;
//        let output_size = self.conf.output_size;
//        let activation = self.conf.activation.unwrap(); //TODO: propagate this with the weird status
//        let input = self.conf.input.as_mut().unwrap().clone();
//
//        let mut scope = scope.new_sub_scope("layer");
//        let scope = &mut scope;
//        let w_shape = ops::constant(&[input_size as i64, output_size as i64][..], scope)?;
//        let w = Variable::builder()
//            .initial_value(
//                ops::RandomStandardNormal::new()
//                    .dtype(DataType::Float)
//                    .build(w_shape.clone(), scope)?,
//            )
//            .data_type(DataType::Float)
//            .shape([input_size, output_size])
//            .build(&mut scope.with_op_name("w"))?;
//
//        let n = ops::constant(input_size as f32, scope)?;
//
//        //NOTE: tan on weights is to force weights to dropout but use the gradient for better dropout than random node based dropout
//        //NOTE: we multiply the activation by 100 to represent values </>than 1/-1
//        let output_op = ops::div(
//            ops::mat_mul(
//                input,
//                //this sets the gradients to dropout weights
//                ops::tan(w.output().clone(), scope)?,
//                scope,
//            )?,
//            n,
//            scope,
//        )?;
//
//        let act = activation
//            .function(
//                //this normalizes to speed up trainning and sample efficiency
//                output_op.into(),
//                scope,
//            )?
//            .clone();
//        //the activatied output signal
//        let output_op = act.clone();
//        // .into(); //,
//        //Ok((vec![w.clone()], act.into(), output_op))
//        let layer = TrainnableLayer {
//            variables: vec![w.clone()],
//            output: act.into(),
//            operation: output_op,
//        };
//        Ok(layer)
//    }
//}

//TODO: uppatch or deprecate the following
// Builder function for norm_layer.
// Passes in layer configuration and returns a pointer to a Layer type.
//pub fn norm() -> Layer {
//    Box::new(norm_layer)
//}
//////// A standard layer with bias term
////////
//////// `activation` is a function which takes a tensor and applies an activation
//////// function such as sigmoid.
////////
//////// Returns variables created and the layer output.
/////fn std_layer<GraphNode: Into<Output>>(
/////    input: GraphNode,
/////    input_size: u64,
/////    output_size: u64,
/////    activation: &dyn Fn(Output, &mut Scope) -> Result<Operation, Status>,
/////    scope: &mut Scope,
/////) -> Result<(Vec<Variable>, Output, Operation), Status> {
/////    let mut scope = scope.new_sub_scope("layer");
/////    let scope = &mut scope;
/////    let w_shape = ops::constant(&[input_size as i64, output_size as i64][..], scope)?;
/////    let mut init_bias: Tensor<f32> = Tensor::new(&[1u64, output_size as u64]);
/////    for i in 0..output_size {
/////        init_bias[i as usize] = 0.0 as f32;
/////    }
/////
/////    let w = Variable::builder()
/////        .initial_value(
/////            ops::RandomStandardNormal::new()
/////                .dtype(DataType::Float)
/////                .build(w_shape, scope)?,
/////        )
/////        .data_type(DataType::Float)
/////        .shape([input_size, output_size])
/////        .build(&mut scope.with_op_name("w"))?;
/////    let b = Variable::builder()
/////        .initial_value(
/////            //this is more sensible than random by the nature of the bias,
/////            //otherwise we will miss correlated signals by default with RandomStandardNormal
/////            ops::constant(
/////                Tensor::new(&[1u64, output_size as u64])
/////                    .with_values(&vec![0.0f32; output_size as usize][..])?,
/////                scope,
/////            )?,
/////            // ops::RandomStandardNormal::new()
/////            //     .dtype(DataType::Float)
/////            //     .build(ops::constant(&[1i64,output_size as i64][..], scope)?, scope)?,
/////        )
/////        .data_type(DataType::Float)
/////        .shape(&[1i64, output_size as i64][..])
/////        .build(&mut scope.with_op_name("b"))?;
/////
/////    //n is input_size to be divided at each node in order to normalize the signals at each node before activation
/////    let act = activation(
/////        ops::add(
/////            ops::mat_mul(input, w.output().clone(), scope)?,
/////            b.output().clone(),
/////            scope,
/////        )?
/////        .into(),
/////        scope,
/////    )?;
/////    let output_op = act.clone();
/////    Ok((vec![w.clone(), b.clone()], act.into(), output_op))
/////}
//////// Builder for fully_connected_layer
//////// Passes in layer configuration and returns a pointer to a Layer type.
/////pub fn fully_connected() -> Layer {
/////    Box::new(std_layer)
/////}
/////
///////TODO: ffs rename this..
//////// A modem layer with bias term that drops out to try to put pressure on corellating signals
//////// demodulates the bias terms signal modulation for better generalization.
////////
//////// `activation` is a function which takes a tensor and applies an activation
//////// function such as sigmoid.
////////
//////// Returns variables created and the layer output.
/////fn modem_layer<GraphNode: Into<Output>>(
/////    input: GraphNode,
/////    input_size: u64,
/////    output_size: u64,
/////    activation: &dyn Fn(Output, &mut Scope) -> Result<Operation, Status>,
/////    scope: &mut Scope,
/////) -> Result<(Vec<Variable>, Output, Operation), Status> {
/////    let mut scope = scope.new_sub_scope("layer");
/////    let scope = &mut scope;
/////    let w_shape = ops::constant(&[input_size as i64, output_size as i64][..], scope)?;
/////    let mut init_bias: Tensor<f32> = Tensor::new(&[1u64, output_size as u64]);
/////    for i in 0..output_size {
/////        init_bias[i as usize] = 0.0 as f32;
/////    }
/////
/////    let w = Variable::builder()
/////        .initial_value(
/////            ops::RandomStandardNormal::new()
/////                .dtype(DataType::Float)
/////                .build(w_shape, scope)?,
/////        )
/////        .data_type(DataType::Float)
/////        .shape([input_size, output_size])
/////        .build(&mut scope.with_op_name("w"))?;
/////    let b = Variable::builder()
/////        .initial_value(
/////            //this is more sensible than random by the nature of the bias,
/////            //otherwise we will miss correlated signals by default with RandomStandardNormal
/////            ops::constant(
/////                Tensor::new(&[1u64, output_size as u64])
/////                    .with_values(&vec![0.0f32; output_size as usize][..])?,
/////                scope,
/////            )?,
/////            // ops::RandomStandardNormal::new()
/////            //     .dtype(DataType::Float)
/////            //     .build(ops::constant(&[1i64,output_size as i64][..], scope)?, scope)?,
/////        )
/////        .data_type(DataType::Float)
/////        .shape(&[1i64, output_size as i64][..])
/////        .build(&mut scope.with_op_name("b"))?;
/////
/////    //n is input_size to be divided at each node in order to normalize the signals at each node before activation
/////    let act = activation(
/////        ops::add(
/////            ops::mat_mul(input, w.output().clone(), scope)?,
/////            // this is only necessary for trainning since it drops out the bias
/////            ops::tanh(b.output().clone(), scope)?,
/////            scope,
/////        )?
/////        .into(),
/////        scope,
/////    )?;
/////    let output_op = act.clone();
/////    Ok((vec![w.clone(), b.clone()], act.into(), output_op))
/////}
//////// Builder for fully_connected_layer
//////// Passes in layer configuration and returns a pointer to a Layer type.
/////pub fn modem() -> Layer {
/////    Box::new(modem_layer)
/////}
/////
/////
///////TODO: @DEPRECATE this since connections are defined with the route passthrough layer in the sequential builder.
/////// if residual connections are to be abstracted, abstract route construction routine. all layers are fully connected
/////// so we only define "node" level operations for simple modularity scaling
////////a normal layer as above but with residual connections
/////fn norm_res_layer<GraphNode: Into<Output>>(
/////    input: GraphNode,
/////    //TODO: trait macro for extra args or a builder with trait?
/////    res_input: GraphNode,
/////    input_size: u64,
/////    output_size: u64,
/////    activation: &dyn Fn(Output, &mut Scope) -> Result<Output, Status>,
/////    scope: &mut Scope,
/////) -> Result<(Vec<Variable>, Output), Status> {
/////    let mut scope = scope.new_sub_scope("layer");
/////    let scope = &mut scope;
/////    let w_shape = ops::constant(&[input_size as i64, output_size as i64][..], scope)?;
/////    let w = Variable::builder()
/////        .initial_value(
/////            ops::RandomStandardNormal::new()
/////                .dtype(DataType::Float)
/////                .build(w_shape.clone(), scope)?,
/////        )
/////        .data_type(DataType::Float)
/////        .shape([input_size, output_size])
/////        .build(&mut scope.with_op_name("w"))?;
/////
/////    //n is input_size to be divided at each node in order to normalize the signals at each node before activation
/////    let input_size = 2 * input_size;
/////    //TODO: concat the input and res tensors
/////    // let concat = ops::concat(0,vec![input, res_input],  scope)?.into();
/////
/////    let scalar_coe = ops::constant(f16::from_f32(0.1), scope)?;
/////    let cur = ops::mat_mul(
/////        input,
/////        //NOTE: division for half stability the higher this value the more stable the trainning but the less expressivity (domain) of the weights
/////        //NOTE: tan on weights is to force weights to dropout but use the gradient for better dropout than random node based dropout
/////        ops::multiply(
/////            ops::tan(w.output().clone(), scope)?,
/////            scalar_coe.clone(),
/////            scope,
/////        )?,
/////        scope,
/////    )?;
/////
/////    // let cur_res = ops::mat_mul(
/////    // res_input,
/////    // ops::multiply(ops::tan(w_res.output().clone(), scope)?, scalar_coe.clone(), scope)?,
/////    // scope,
/////    // )?;
/////    // ops::tan(w_res.output().clone(), scope)?, scope)?;
/////    let res_input = ops::multiply(scalar_coe, res_input, scope)?;
/////    let cur = ops::add(cur, res_input, scope)?;
/////
/////    let n = ops::constant(f16::from_f32(input_size as f32), scope)?;
/////
/////    let res = activation(ops::div(cur, n, scope)?.into(), scope)?.into(); //,
/////
/////    // Ok((vec![w.clone(), w_res.clone()], res))
/////    Ok((vec![w.clone()], res))
/////}
/////
///////TODO: feedback network with parameter for depth of hidden layer to feedback_input_vector
/////
///////test for merge and fork
/////#[test]
/////fn test_concat_split() {
/////    println!("test_concat_split");
/////    let mut scope = Scope::new_root_scope();
/////    println!("create a tensor");
/////    let mut t: Tensor<f32> = Tensor::new(&[1, 6][..]);
/////    //push 123456 into t
/////    for i in 0..6 {
/////        t[i] = i as f32;
/////    }
/////    println!("created tensor {:?}", t);
/////    for i in 0..6 {
/////        println!("{}", t[i]);
/////    }
/////    println!("create a constant");
/////    let input = ops::constant(t.clone(), &mut scope).unwrap();
/////    println!("created constant {:?}", t);
/////    println!("split the tensor");
/////    let split_outputs = split((input.clone().into(), input.clone()), 2 as i32, &mut scope).unwrap();
/////    // println!("merge the tensors");
/////    println!("merge the tensors {:?}", split_outputs.clone());
/////    let merged_output = merge(split_outputs.clone(), &mut scope).unwrap();
/////
/////    let session = Session::new(&SessionOptions::new(), &mut scope.graph()).unwrap();
/////    let mut run_args = SessionRunArgs::new();
/////    let merged_outputs_fetch = run_args.request_fetch(&merged_output.operation, 0);
/////
/////    println!("length of split outputs: {}", split_outputs.len());
/////    for i in 0..2 {
/////        let split_output = split_outputs[i].clone();
/////        println!("{:?}", split_output.name());
/////    }
/////
/////    session.run(&mut run_args).unwrap();
/////    let merged_output: Tensor<f32> = run_args.fetch(merged_outputs_fetch).unwrap();
/////    //TODO: may need to reshape here getting shape: [2, 3]
/////    println!("merged output: {:?}", merged_output);
/////    println!("merged output shape: {:?}", merged_output.shape());
/////    //NOTE: iter impl apparently is awesome and goes row then column wise flattened
/////    println!("merged output len: {:?}", merged_output.len());
/////    for i in 0..6 {
/////        println!("{}", merged_output[i]);
/////    }
/////}
/////
///////TODO: test drive this development to builder extraction.
/////#[test]
/////fn test_fully_connected_tower() {
/////    // use crate::activations::tanh;
/////    use crate::activations::relu;
/////    println!("test_fully_connected");
/////    let mut scope = Scope::new_root_scope(); //.with_device("/device:gpu:0");
/////
/////    //BUILD THE INPUT
/////    //TODO: START OF EXTRACTION
/////    println!("create a tensor");
/////    let mut t: Tensor<f32> = Tensor::new(&[1, 40][..]);
/////    for i in 0..40 {
/////        t[i] = i as f32;
/////    }
/////    println!("created tensor {:?}", t);
/////    for i in 0..40 {
/////        println!("{}", t[i]);
/////    }
/////    println!("create a constant");
/////    //create a label tensor just like above but set values to 1
/////    let mut l: Tensor<f32> = Tensor::new(&[1, 40][..]);
/////    for i in 0..40 {
/////        l[i] = 1.0 as f32;
/////    }
/////
/////    //TODO: END OF EXTRACTION
/////
/////    let input = ops::Placeholder::new()
/////        .dtype(DataType::Float)
/////        .shape(&[1u64, 40])
/////        .build(&mut scope.with_op_name("Input"))
/////        .unwrap();
/////    let label = ops::Placeholder::new()
/////        .dtype(DataType::Float)
/////        //TODO: output size
/////        .shape(&[1u64, 40])
/////        .build(&mut scope.with_op_name("Label"))
/////        .unwrap();
/////
/////    // let input = ops::constant(t, &mut scope).unwrap();
/////
/////    //BUILD THE NETWORK
/////    println!("split the tensor");
/////    let split_outputs = split((input.clone().into(), input.clone()), 4 as i32, &mut scope).unwrap();
/////    //now create a std_layer for each split
/////    let mut layers = vec![];
/////    for i in 0..4 {
/////        let (w, b, output) =
/////            std_layer(split_outputs[i].to_owned(), 10, 10, &relu(10), &mut scope).unwrap();
/////        layers.push((w, b, output));
/////    }
/////    //TODO: this probably will need to be abstracted in builder
/////    let layer_outputs: Vec<Output> = layers
/////        .iter()
/////        .map(|(w, b, output)| {
/////            let res: Output = output.clone().into();
/////            res
/////        })
/////        .collect::<Vec<Output>>();
/////    println!("merge the std_layer tensors {:?}", layer_outputs);
/////    let merged_output = merge(layer_outputs.clone(), &mut scope).unwrap();
/////    //create one more std_layer
/////    let final_output = std_layer(merged_output, 40, 40, &relu(10), &mut scope).unwrap();
/////    // layer_outputs.push(final_output.1.clone());
/////    layers.push(final_output.clone());
/////
/////    //PREPARE OPTIMIZER
/////    let mut optimizer = AdadeltaOptimizer::new();
/////    optimizer.set_learning_rate(ops::constant(0.01 as f32, &mut scope).unwrap());
/////
/////    let Output = final_output.1;
/////    let Output_op = final_output.2;
/////
/////    //PREPARE VARIABLES AND ERROR FUNCTION
/////    let mut vars = vec![];
/////    for layer in layers.into_iter().map(|(v, act, output)| v) {
/////        for var in layer {
/////            println!("{:?}", var.name());
/////            vars.push(var);
/////        }
/////    }
/////
/////    //TODO: Error function
/////    let Error = ops::sqrt(
/////        ops::pow(
/////            ops::sub(Output.clone(), label.clone(), &mut scope).unwrap(),
/////            ops::constant(2.0 as f32, &mut scope).unwrap(),
/////            &mut scope,
/////        )
/////        .unwrap(),
/////        &mut scope.with_op_name("Error"),
/////    )
/////    .unwrap();
/////    // let Error = ops::pow(
/////    //     Error.clone(),
/////    //     ops::constant(7.0 as f32, &mut scope).unwrap(),
/////    //     &mut scope.with_op_name("error"),
/////    // ).unwrap();
/////
/////    let (minimize_vars, minimize) = optimizer
/////        .minimize(
/////            &mut scope,
/////            Error.clone().into(),
/////            MinimizeOptions::default().with_variables(&vars),
/////        )
/////        .unwrap();
/////
/////    for var in minimize_vars {
/////        println!("{:?}", var.name());
/////        vars.push(var);
/////    }
/////    //RUN SESSION
/////    //this should probably be extracted to an initialization routine like in Brains since may be needed in other places
/////    let session = Session::new(&SessionOptions::new(), &mut scope.graph()).unwrap();
/////    let mut run_args = SessionRunArgs::new();
/////
/////    //initialize the variables
/////    for var in vars.iter() {
/////        run_args.add_target(&var.initializer());
/////    }
/////    //TODO: add input and label feed and actually propagate.
/////    session.run(&mut run_args).unwrap();
/////
/////    let mut merged_output: Option<Tensor<f32>> = None;
/////    let mut final_error: Option<Tensor<f32>> = None;
/////
/////    for  in 0..100000 {
/////        let mut run_args = SessionRunArgs::new();
/////        run_args.add_target(&minimize);
/////        let fetch_error = run_args.request_fetch(&Error, 0);
/////        //TODO: can add_feed use new inst feature paradigm?
/////        run_args.add_feed(&input, 0, &t);
/////        run_args.add_feed(&label, 0, &l);
/////
/////        let merged_outputs_fetch = run_args.request_fetch(&Output_op, 0);
/////        session.run(&mut run_args).unwrap();
/////
/////        // let merged_output: Tensor<f32> = run_args.fetch(merged_outputs_fetch).unwrap();
/////        merged_output = Some(run_args.fetch(merged_outputs_fetch).unwrap());
/////        // println!("merged output: {:?}", merged_output);
/////        //print fetch error
/////        // let error:Tensor<f32> = run_args.fetch(fetch_error).unwrap();
/////        final_error = Some(run_args.fetch(fetch_error).unwrap());
/////        // for i in 0..40 {
/////        //     println!("{}", error[i]);
/////        // }
/////    }
/////    for i in 0..40 {
/////        println!("{}", merged_output.clone().unwrap()[i]);
/////    }
/////    //print this is error:
/////    println!("this is error:");
/////    for i in 0..40 {
/////        println!("{}", final_error.clone().unwrap()[i]);
/////    }
/////}
/////
/////// TODO: Unimplemented/Unanalyzed but interesting to offload some of the stuff from tf.nn to
/////// native rust
////////Merge layers take in a vector of layers and concatenates them into a single layer.
////////
//////// To convolve, split into overlap_size chunks and send overlap chunks with the non
//////// overlapped chunks to each merge operation.
/////fn merge(inputs: Vec<Output>, scope: &mut Scope) -> Result<Output, Status> {
/////    let zero: Output = ops::constant(1, scope)?.into();
/////    println!("merge inputs: {:?}", inputs.len());
/////    let group_op = ops::ConcatV2::new().build_instance(inputs.clone(), zero, scope)?;
/////    Ok(group_op.output())
/////}
/////
////////Split layers take in a single layer and slice it into a vector of layers. Since all layers are 1 dimensional,
////////we can slice with a single num_splits on axis=0. This is invariant to layer width so it can be scaled.
////////
//////// this doesnt return Operations since this is supposed to be a transparent connection only operation.
/////fn split(
/////    input: (Output, Operation),
/////    num_splits: i32,
/////    scope: &mut Scope,
/////) -> Result<Vec<Output>, Status> {
/////    //assert length is divisible by num_splits, since we are only constructing a graph this should happen in
/////    //release as well
/////    let axis_one: Output = ops::constant(1, scope)?.into();
/////
/////    let split_operation =
/////        ops::Split::new()
/////            .num_split(num_splits)
/////            .build_instance(axis_one.clone(), input.0, scope)?;
/////
/////    //print the number of outputs:
/////    let split_outputs = split_operation.output()?;
/////    Ok(split_outputs.into())
/////}
