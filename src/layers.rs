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
use InheritDerive::InheritState;

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
    fn get_activation(&self) -> &Activation;
    fn get_dtype(&self) -> DataType;
}
///A trait that defines the standard layer parameters via getters
///and setters that can mutably configure layers, used by internal network builder.
pub trait ConfigureLayer {
    fn input(self, input: Operation) -> Self;
    fn input_size(self, input_size: u64) -> Self;
    fn output_size(self, output_size: u64) -> Self;
    fn width(self, width: u64) -> Self;
    fn activation(self, activation: Activation) -> Self;
    fn dtype(self, dtype: DataType) -> Self;
}
pub trait ConfigureLayerNoChain {
    fn input(&mut self, input: Operation);
    fn input_size(&mut self, input_size: u64);
    fn output_size(&mut self, output_size: u64);
    fn width(&mut self, width: u64);
    fn activation(&mut self, activation: Activation);
    fn dtype(&mut self, dtype: DataType);
}

///Layer constructor trait with default values.
pub trait InitializeLayer {
    fn init() -> Self
    where
        Self: Sized;
}
///Set all user defined parameters for a layer.
pub trait build_layer {
    fn layer(width: u64, activation: Activation) -> Box<Self>
    where
        Self: Sized;
}
impl<T> build_layer for T
where
    T: InitializeLayer + ConfigureLayer,
{
    fn layer(width: u64, activation: Activation) -> Box<Self> {
        Box::new(Self::init().width(width).activation(activation))
    }
}
pub trait Layer: BuildLayer + AccessLayer + ConfigureLayerNoChain + InheritState {}
impl<L> Layer for L where L: BuildLayer + AccessLayer + ConfigureLayerNoChain + InheritState {}

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
    activation: Activation,
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
    fn activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }
    fn dtype(mut self, dtype: DataType) -> Self {
        self.dtype = dtype;
        self
    }
}
impl ConfigureLayerNoChain for LayerState {
    fn input(&mut self, input: Operation) {
        self.input = Some(input);
    }
    fn input_size(&mut self, input_size: u64) {
        self.input_size = input_size;
    }
    fn output_size(&mut self, output_size: u64) {
        self.output_size = output_size;
    }
    fn width(&mut self, width: u64) {
        self.width = width;
    }
    fn activation(&mut self, activation: Activation) {
        self.activation = activation;
    }
    fn dtype(&mut self, dtype: DataType) {
        self.dtype = dtype;
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
    fn get_activation(&self) -> &Activation {
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
    fn activation(mut self, activation: Activation) -> Self {
        self.get_mut_layer_state().activation = activation;
        self
    }
    fn dtype(mut self, dtype: DataType) -> Self {
        self.get_mut_layer_state().dtype = dtype;
        self
    }
}
impl<L> ConfigureLayerNoChain for L
where
    L: InheritState,
{
    fn input(&mut self, input: Operation) {
        self.get_mut_layer_state().input = Some(input);
    }
    fn input_size(&mut self, input_size: u64) {
        self.get_mut_layer_state().input_size = input_size;
    }
    fn output_size(&mut self, output_size: u64) {
        self.get_mut_layer_state().output_size = output_size;
    }
    fn width(&mut self, width: u64) {
        self.get_mut_layer_state().width = width;
    }
    fn activation(&mut self, activation: Activation) {
        self.get_mut_layer_state().activation = activation;
    }
    fn dtype(&mut self, dtype: DataType) {
        self.get_mut_layer_state().dtype = dtype;
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
    fn get_activation(&self) -> &Activation {
        &self.get_layer_state().activation
    }
    fn get_dtype(&self) -> DataType {
        self.get_layer_state().dtype
    }
}

//LAYER DEFINITIONS//
//Layers use an optional inheritence pattern for fast blanket implementation
//generalization via accessors to define shared structure.
//each layer doesnt have to inherit LayerState and derive InheritState but you will
//likely end up doing this yourself otherwise.
//NOTE: currently AccessLayer and ConfigureLayer are manditory for integrating into Brains so the
//optional inheritence pattern is *currently* the standard for extensibility
#[derive(InheritState)]
pub struct fully_connected(LayerState);
impl BuildLayer for fully_connected {
    fn build_layer(&self, scope: &mut Scope) -> Result<TrainnableLayer, Status> {
        //instead use the accessor
        let input_size = self.get_input_size();
        let output_size = self.get_output_size();
        let input = self.get_input().clone().unwrap();
        let dtype = self.get_dtype();
        let mut scope = scope.new_sub_scope("std_layer"); //TODO: layer counter or some uuid?
        let scope = &mut scope;
        let w_shape = ops::constant(&[input_size as i64, output_size as i64][..], scope)?;
        let mut init_bias: Tensor<f32> = Tensor::new(&[1u64, output_size as u64]);
        for i in 0..output_size {
            init_bias[i as usize] = 0.0 as f32;
        }

        let w = Variable::builder()
            .initial_value(
                ops::RandomStandardNormal::new()
                    .dtype(dtype)
                    .build(w_shape, scope)?,
            )
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
                )?,
            )
            .data_type(dtype)
            .shape(&[1i64, output_size as i64][..])
            .build(&mut scope.with_op_name("b"))?;

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
            variables: vec![w, b],
            output: act.into(),
            operation: output_op,
        })
    }
}

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
/// the activation function should be bounded -1 > x > 1
///
/// NOTE:
/// We dont use BFloat since the integer range is only used as a buffer for addition overflow in matmul.
/// In all other operations we are strictly bounded -1 > x > 1. As long as layer_width is not
/// greater than Float range we are fine in the worst case (summing all 1's).
/// Otherwise decimal precision of float type is our parameter type precision.
#[derive(InheritState)]
//pub struct norm {
//    pub layer_state: LayerState,
//}
pub struct norm(LayerState);
impl BuildLayer for norm {
    fn build_layer(&self, scope: &mut Scope) -> Result<TrainnableLayer, Status> {
        //TODO: this solves exploding gradient but how can we fix vanishing gradient?
        //instead use the accessor
        let input_size = self.get_input_size();
        let output_size = self.get_output_size();
        let input = self.get_input().clone().unwrap();
        let dtype = self.get_dtype();
        let mut scope = scope.new_sub_scope("norm_layer"); //TODO: layer counter or some uuid?
        let scope = &mut scope;
        let w_shape = ops::constant(&[input_size as i64, output_size as i64][..], scope)?;

        let w = Variable::builder()
            .initial_value(
                ops::RandomStandardNormal::new()
                    .dtype(dtype)
                    .build(w_shape, scope)?,
            )
            .data_type(dtype)
            .shape([input_size, output_size])
            .build(&mut scope.with_op_name("w"))?;

        //TODO: this should be a 0 rank const
        let input_size = Tensor::new(&[1u64, output_size as u64])
            .with_values(&vec![input_size; output_size as usize][..])?;
        let input_size_const = ops::constant(input_size, scope)?;
        let input_size_as_dtype = ops::Cast::new()
            .SrcT(DataType::UInt64)
            .DstT(dtype.clone())
            .build(input_size_const, scope)?;

        //take w.output() and set an artificial gradient of x^2, this causes a normalized gradient that
        //goes to 0 as the weights go to 0.
        let three = ops::constant(3, scope)?;
        let three = ops::Cast::new()
            .SrcT(DataType::Int32)
            .DstT(dtype.clone())
            .build(three.clone(), scope)?;

        let dropout_derivative = ops::div(
            //mul by itself 3 times
            //ops::mul(w.output().clone(), three.clone(), scope)?,
            ops::mul(
                ops::mul(w.output().clone(), w.output().clone(), scope)?,
                w.output().clone(),
                scope,
            )?,
            three,
            scope,
        )?;

        let act = (self.get_activation().as_ref().unwrap())(
            ops::div(
                //use a tangent operation to cause connection wise dropout. we can do this because
                //the paramaters and signals are normalized.
                ops::mat_mul(input.clone(), dropout_derivative, scope)?,
                input_size_as_dtype,
                scope,
            )?
            .into(),
            scope,
        )?;

        let output_op = act.clone();
        Ok(TrainnableLayer {
            variables: vec![w],
            output: act.into(),
            operation: output_op,
        })
    }
}
//TODO: scale layer, just multiplies by a constant
///scale
#[derive(InheritState)]
pub struct scale {
    pub layer_state: LayerState,
    pub scale: f32,
}
impl BuildLayer for scale {
    fn build_layer(&self, scope: &mut Scope) -> Result<TrainnableLayer, Status> {
        let input = self.get_input().clone().unwrap();
        let dtype = self.get_dtype();
        let mut scope = scope.new_sub_scope("scale_layer"); //TODO: layer counter or some uuid?
        let scope = &mut scope;
        let scale = ops::constant(self.scale, scope)?;
        let scale = ops::Cast::new()
            .SrcT(DataType::Float)
            .DstT(dtype.clone())
            .build(scale, scope)?;
        let scale_op = ops::mul(input.clone(), scale, scope)?;
        let output_op = scale_op.clone();
        Ok(TrainnableLayer {
            variables: vec![],
            output: scale_op.into(),
            operation: output_op,
        })
    }
}
impl scale {
    pub fn scale(mut self, scale: f32) -> Box<Self> {
        self.scale = scale;
        Box::new(self)
    }
    pub fn magnitude(scale: f32) -> Box<Self> {
        scale::layer(0, None).scale(scale)
    }
}
