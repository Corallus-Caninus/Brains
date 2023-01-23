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
use rand::Rng;
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
///accessors for shared state of a layer
pub trait AccessLayer {
    fn get_input(&self) -> &Option<Operation>;
    fn get_input_size(&self) -> u64;
    fn get_output_size(&self) -> u64;
    fn get_width(&self) -> u64;
    fn get_activation(&self) -> &Activation;
    fn get_dtype(&self) -> DataType;
}
///setters for shared state of a layer that can be method chained
pub trait ConfigureLayer {
    fn input(self, input: Operation) -> Self;
    fn input_size(self, input_size: u64) -> Self;
    fn output_size(self, output_size: u64) -> Self;
    fn width(self, width: u64) -> Self;
    fn activation(self, activation: Activation) -> Self;
    fn dtype(self, dtype: DataType) -> Self;
}
///Setters for shared state of a layer that dont return self
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
//TODO: resolve ConfigureLayer vs ConfigureLayerNoChain with a blanket impl
///master trait for traits that define a layers required methods
pub trait Layer: BuildLayer + AccessLayer + ConfigureLayerNoChain + InheritState {}
impl<L> Layer for L where L: BuildLayer + AccessLayer + ConfigureLayerNoChain + InheritState {}

//FUNDAMENTAL LAYER STATE//
///A subset of concrete state standardized to define any layer in tf. Used as a foundation for
///adding new layers to the framework and inherently derives accessors and setters.
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
///Operation ready to be assigned and trainnable variables (parameters) exposed.
pub struct TrainnableLayer {
    pub variables: Vec<Variable>,
    pub output: Output,
    pub operation: Operation,
}

///process a type, adding it to the graph and returning a TrainnableLayer
pub trait BuildNetwork {
    fn build(self, Scope: &mut Scope) -> Result<Vec<TrainnableLayer>, Status>;
}
//TODO: this should be a derive macro for any T | T.0 = LayerState
///accessors to get the underlying inherited LayerState
pub trait InheritState {
    fn get_mut_layer_state(&mut self) -> &mut LayerState;
    fn get_layer_state(&self) -> &LayerState;
}
//blanket for optional inherited layer state
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
///a fully connected layer
#[derive(InheritState)]
pub struct fully_connected(LayerState);
impl BuildLayer for fully_connected {
    fn build_layer(&self, scope: &mut Scope) -> Result<TrainnableLayer, Status> {
        //instead use the accessor
        let input_size = self.get_input_size() as i32;
        let output_size = self.get_output_size() as i32;
        let input = self.get_input().clone().unwrap();
        let dtype = self.get_dtype();
        let mut scope = scope.new_sub_scope("std_layer"); //TODO: layer counter or some uuid?
        let scope = &mut scope;
        let w_shape = ops::constant(&[input_size, output_size][..], scope)?;
        let mut init_bias: Tensor<f32> = Tensor::new(&[1, output_size as u64]);
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
                        Tensor::new(&[1 as u64, output_size as u64])
                            .with_values(&vec![0.0f32; output_size as usize][..])?,
                        scope,
                    )?,
                    scope,
                )?,
            )
            .data_type(dtype)
            .shape(&[1, output_size][..])
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

///a fully connected layer with zero initialized weights (to allow all ops to run on GPU since
///initializers complain that they require CPU device with scope.device("/gpu:0") )
#[derive(InheritState)]
pub struct fully_connected_zero_init(LayerState);
impl BuildLayer for fully_connected_zero_init {
    fn build_layer(&self, scope: &mut Scope) -> Result<TrainnableLayer, Status> {
        //instead use the accessor
        let input_size = self.get_input_size() as i32;
        let output_size = self.get_output_size() as i32;
        let input = self.get_input().clone().unwrap();
        let dtype = self.get_dtype();
        let mut scope = scope.new_sub_scope("std_layer_zero_init"); //TODO: layer counter or some uuid?
        let scope = &mut scope;
        let w_shape = ops::constant(&[input_size, output_size][..], scope)?;
        let mut init_bias: Tensor<f32> = Tensor::new(&[1, output_size as u64]);
        for i in 0..output_size {
            init_bias[i as usize] = 0.0 as f32;
        }

        let init_w = ops::constant(
            Tensor::new(&[input_size as u64, output_size as u64])
                //.with_values(&vec![0.0f32; (input_size * output_size) as usize][..])?,
                //same as above but use dtype
                .with_values(&vec![0.0f32; (input_size * output_size) as usize][..])?,
            scope,
        )?;
        let w = Variable::builder()
            .initial_value(
                ops::Cast::new()
                    .SrcT(DataType::Float)
                    .DstT(dtype)
                    .build(init_w, scope)?,
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
                        Tensor::new(&[1 as u64, output_size as u64])
                            .with_values(&vec![0.0f32; output_size as usize][..])?,
                        scope,
                    )?,
                    scope,
                )?,
            )
            .data_type(dtype)
            .shape(&[1, output_size][..])
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
#[derive(InheritState)]
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
        let w_shape = ops::constant(&[input_size as i32, output_size as i32][..], scope)?;

        let w = Variable::builder()
            .initial_value(
                ops::RandomStandardNormal::new()
                    .dtype(dtype)
                    .build(w_shape, scope)?,
            )
            .data_type(dtype)
            .shape([input_size, output_size])
            .build(&mut scope.with_op_name("w"))?;

        //TODO: this should be a 0 rank const (but that would just be a broadcast anyways)
        //let input_size = Tensor::new(&[1u64, output_size as u64])
        //    .with_values(&vec![input_size; output_size as usize][..])?;
        let input_size_const = ops::constant(input_size, scope)?;
        let input_size_as_dtype = ops::Cast::new()
            .SrcT(DataType::UInt64)
            .DstT(dtype.clone())
            .build(input_size_const, scope)?;

        let act = (self.get_activation().as_ref().unwrap())(
            ops::div(
                ops::mat_mul(input.clone(), w.output().clone(), scope)?,
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
//TODO: should we have a layer that doesnt activate instead and let this be learned by weights? can
//just add another layer or a conditional builder for layers (is_last: bool)
///A layer that multiplies each entry in the previous Tensor by a constant, usually a multiple of
///10. This allows last layer activation to express values beyond squashing where necessory
/// (sigmoid, tanh, and all other bounded activation functions.)
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
