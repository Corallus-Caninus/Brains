//TODO: floats stink norm net is an attempt to realizing rot net and uint8 operations with bounded functions while retaining all the flaws in DNNs.. (go back to rot net at some point)
//TODO: extract layers to layers.rs and upload to crates.io with example of how to implement an architecture

//TODO: type errors for anything other than u64 architecture due to casts.
//      Also precision errors for summing f32s which should accumulate to f64.
//TODO: logging for effeciency, SED log::debug! to a logging crate since stdout logs are already formatted well
//NOTE: DONE WITH FEATURES. this is a basic layers based tensorflow backed ANN architecture framework.
//      Anything else should be written outside this crate and integrated only if general enough.

//TODO: reflection on the graph for parameter count, total KB/MB/GB/TB data trained on, cross validated accuracy avg
//      for trained data, where the most recent serialization has been saved etc.

//allow unstable features
// #![feature(int_log)]

mod activations;
mod layers;
mod optimizer;
use anyhow::Context;
use optimizer::LossFunction;
//use half::bf16;
//use half::f16;
use log;
use rand::seq::IteratorRandom;
use serde::{Deserialize, Serialize};
use serde_derive::{Deserialize, Serialize};
use serde_json::Value;
use std::cell::RefCell;
use std::env;
use std::error::Error;
use std::fs;
use std::io::ErrorKind;
use std::io::{Read, Write};
use std::path::Path;
use std::result::Result;
//TODO: clean this up with proper heirachy
use rand::Rng;
// include par_iter from rayon
use layers::*;
use rayon::iter::ParallelBridge;
use rayon::prelude::*;
use std::collections::HashMap;
use std::os;
use std::rc::Rc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tensorflow::ops;
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
use tensorflow::Variable;
use tensorflow::REGRESS_INPUTS;
use tensorflow::REGRESS_METHOD_NAME;
use tensorflow::REGRESS_OUTPUTS;
use uuid::Uuid;
extern crate tensorflow;

//PATCH FIXES THAT ARE TODO: FIX UPSTREAM TF-rs//
//TODO: fix this:
//unsafe impl<T> Send for Tensor<T> {}
//create an extension struct just to derive the above impl:
pub struct SendableTensor<T>
where
    T: tensorflow::TensorType,
    <T as tensorflow::TensorType>::InnerType: Send + Sync,
{
    pub tensor: Tensor<T>,
}
unsafe impl<T> Send for SendableTensor<T>
where
    T: tensorflow::TensorType,
    <T as tensorflow::TensorType>::InnerType: Send + Sync,
{
}
impl<T> SendableTensor<T>
where
    T: tensorflow::TensorType,
    <T as tensorflow::TensorType>::InnerType: Send + Sync,
{
    pub fn new(tensor: Tensor<T>) -> SendableTensor<T> {
        SendableTensor { tensor }
    }
}
impl<T> From<Tensor<T>> for SendableTensor<T>
where
    T: tensorflow::TensorType,
    <T as tensorflow::TensorType>::InnerType: Send + Sync,
{
    fn from(tensor: Tensor<T>) -> SendableTensor<T> {
        SendableTensor { tensor }
    }
}
impl<T> Into<Tensor<T>> for SendableTensor<T>
where
    T: tensorflow::TensorType,
    <T as tensorflow::TensorType>::InnerType: Send + Sync,
{
    fn into(self) -> Tensor<T> {
        self.tensor
    }
}
impl<T> Clone for SendableTensor<T>
where
    T: tensorflow::TensorType,
    <T as tensorflow::TensorType>::InnerType: Send + Sync,
{
    fn clone(&self) -> Self {
        SendableTensor {
            tensor: self.tensor.clone(),
        }
    }
}
//END OF PATCH FIXES//

//create a trait for BuildLayer + AccessLayer + ConfigureLayer
//TODO: we use NoChain since dyn cant return Self
pub struct BrainBuilder<'a> {
    //Layer: BuildLayer + AccessLayer + ConfigureLayerNoChain> {
    name: &'a str,
    num_inputs: u64,
    dtype: DataType,
    Optimizer: Option<optimizer::LossOptimizer>,
    error: Option<LossFunction>,
    layers: Vec<Box<dyn Layer>>,
}
///Constructor to initialize the BrainBuilder
pub fn Brain<'a>() -> BrainBuilder<'a> {
    BrainBuilder {
        name: "Brain",
        num_inputs: 0,
        dtype: DataType::Float,
        Optimizer: None,
        error: None,
        layers: Vec::new(),
    }
}
//TODO: a trait or some way to do std_layer instead of std_layer::new() (maybe)
impl<'a> BrainBuilder<'a>
//Opt: Optimizer,
{
    pub fn add_layer(mut self, layer: Box<dyn Layer>) -> Self {
        self.layers.push(layer);
        self
    }
    pub fn name(mut self, name: &'a str) -> Self {
        self.name = name;
        self
    }
    pub fn num_inputs(mut self, num_inputs: u64) -> Self {
        self.num_inputs = num_inputs;
        self
    }
    pub fn dtype(mut self, dtype: DataType) -> Self {
        self.dtype = dtype;
        self
    }
    pub fn optimizer(mut self, optimizer: optimizer::LossOptimizer) -> Self {
        self.Optimizer = Some(optimizer);
        self
    }
    pub fn error(mut self, error: LossFunction) -> Self {
        self.error = Some(error);
        self
    }
    pub fn build(mut self, scope: &mut Scope) -> Result<Brain, Status> {
        //TODO: ~Just Rust Things~
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
        for i in 0..self.layers.len() {
            layers.push(self.layers.pop().unwrap());
        }
        layers.reverse();

        return Brain::new(
            self.name.to_string(),
            layers,
            self.Optimizer,
            self.error,
            self.num_inputs,
            self.dtype,
            scope,
        );
    }
}

//TODO: what else should be exposed as brain state? e.g.: dtype?
pub struct Brain<'a> {
    /// Tensorflow objects for user abstraction from Tensorflow
    scope: &'a mut Scope,
    session: Session,
    session_options: SessionOptions,
    ///The error function for the optimizer
    error: Operation,
    ///The DataType for the network operations
    dtype: DataType,
    ///Each layers output for the network
    net_layers: Vec<Output>,
    ///All the trainable parameters of the network
    net_vars: Vec<Variable>,
    ///Operation hooks to interact with the graph (kept here for serialization)
    Input: Operation,
    Label: Operation,
    Output_op: Operation,
    ///variables to be minimized
    minimize_vars: Vec<Variable>,
    ///regression operation
    minimize: Operation,
    ///class for serializing, saving and loading the model
    SavedModelSaver: RefCell<Option<SavedModelSaver>>,
    ///user specified name of this model
    name: String,
}
impl<'a> Brain<'a> {
    //TODO: type safety: use trait bounds to allow for using bigints etc for counting//indexing
    //      types.
    pub fn new(
        name: String,
        //TODO: a vec of layers here will suffice for now, but this will be a builder pattern as
        //soon as possible
        layers: Vec<Box<dyn Layer>>,
        //TODO: optimizer,
        optimizer: Option<optimizer::LossOptimizer>,
        error: Option<LossFunction>,
        num_inputs: u64,
        dtype: DataType,
        scope: &mut Scope,
    ) -> Result<Brain, Status> {
        //propagate an error if optimizer is none
        let optimizer = optimizer.unwrap();
        let error = error.unwrap();

        let mut scope = scope;

        let mut input_size = num_inputs;
        let mut output_size = layers[0].get_width();
        //TODO: may want more than a vector rank dimension, use this with an axis
        //unroll to batch in inputs therefor: +1 Rank
        let Input = ops::Placeholder::new()
            //.dtype(DataType::Float)
            .dtype(dtype)
            .shape([1u64, input_size])
            .build(&mut scope.with_op_name("input"))?;
        let Label = ops::Placeholder::new()
            //.dtype(DataType::Float)
            .dtype(dtype)
            .shape([1u64, layers[layers.len() - 1].get_width()])
            .build(&mut scope.with_op_name("label"))?;

        //CONSTRUCT NETWORK
        //TODO: extract this to a builder pattern for constructor readability.

        //trainnable variables
        let mut net_vars = vec![];
        //each layer as a TF Operation
        let mut net_layers = vec![];

        let mut layer_input_iter = Input.clone();
        for mut layer in layers {
            //TODO: assign Input to each layer. extract this to builder later?
            //TODO: this should already be done in layer builder
            //      (not the aformbentioned Brain builder)

            //set the dtype based on the Brain dtype
            layer.dtype(dtype);
            layer.input(layer_input_iter.clone());
            layer.input_size(input_size);
            input_size = layer.get_width();
            output_size = input_size;
            layer.output_size(output_size);

            let parameters = layer.build_layer(&mut scope)?;
            let vars = parameters.variables;
            let output = parameters.output;

            net_vars.extend(vars);
            net_layers.push(output.clone());

            layer_input_iter = parameters.operation;
            println!("layer: {:?}", layer_input_iter);
            println!("layer_width: {:?}", layer.get_width());
        }
        //TODO: off by 1 at output
        let Output = net_layers.last().unwrap().clone();
        let Output_op = Output.operation.clone();

        let options = SessionOptions::new();
        let SavedModelSaver = RefCell::new(None);

        //TODO: pass this in? this should be modular so new implementations can be tried
        //let mut optimizer = AdadeltaOptimizer::new();
        //TODO: extract this default to Builder

        let error = error(scope, &Output, &Label)?;
        let (minimize_vars, minimize) = optimizer
            .minimize(
                &mut scope,
                error.clone().into(),
                MinimizeOptions::default().with_variables(&net_vars),
            )?
            .into();
        let session = Session::new(&options, &mut scope.graph())?;

        // set parameters to be optimization targets if they havent been set already
        let mut run_args = SessionRunArgs::new();
        for var in &net_vars {
            run_args.add_target(&var.initializer());
        }
        for var in &minimize_vars {
            run_args.add_target(&var.initializer());
        }
        session.run(&mut run_args)?;

        let mut init_brain = Brain {
            name,
            scope,
            session,
            session_options: options,
            dtype,
            net_layers,
            net_vars,
            Input,
            Label,
            Output_op,
            error,
            minimize_vars,
            minimize,
            SavedModelSaver,
        };

        Ok(init_brain)
    }

    ///save all Brain state
    fn serialize_network(&self, dir: String) -> Result<(), Box<dyn Error>> {
        let name = self.name.clone();
        // create a serialized_network object
        let serialized_network = SerializedNetwork {
            parent_search_name: name.to_string(),
        };

        //save the parent_search_name to dir which is where the model is saved, this is the edge to the checkpoint tree
        let file_name = format!("{}/native_rust_data.json", dir);
        log::debug!("serializing non-tensorflow graph variables: {}", file_name);
        // create the file
        let mut file = fs::File::create(file_name.clone())?;
        let serialized_network_string = serde_json::to_string(&serialized_network)?;
        // open the file and write name to it
        file.write_all(serialized_network_string.as_bytes())?;
        file.sync_all()?;

        Ok(())
    }

    //TODO: should we just be passing in the save target here? dir?
    ///Save the model out to disk in this directory as default
    pub fn save(&mut self, name: &str) -> Result<(), Box<dyn Error>> {
        // save the model to disk in the current directory
        if self.SavedModelSaver.borrow().is_none() {
            log::debug!("initializing saved model saver..");
            let mut all_vars = self.net_vars.clone();
            all_vars.extend_from_slice(&self.minimize_vars);
            let mut builder = tensorflow::SavedModelBuilder::new();
            builder
                .add_collection("train", &all_vars)
                .add_tag("serve")
                .add_tag("train")
                .add_signature(REGRESS_METHOD_NAME, {
                    let mut def = SignatureDef::new(REGRESS_METHOD_NAME.to_string());
                    def.add_input_info(
                        REGRESS_INPUTS.to_string(),
                        TensorInfo::new(
                            //DataType::Float,
                            self.dtype,
                            Shape::from(None),
                            OutputName {
                                name: self.Input.name()?,
                                index: 0,
                            },
                        ),
                    );
                    def.add_input_info(
                        "label".to_string(),
                        TensorInfo::new(
                            //DataType::Float,
                            self.dtype,
                            Shape::from(None),
                            OutputName {
                                name: self.Label.name()?,
                                index: 0,
                            },
                        ),
                    );
                    //NOTE: we only need this for reporting learning position for learning rate
                    //TODO: should this be add_output_info since optimizer is already built? I
                    //think this is correct but it's been awhile since I looked at this code.
                    def.add_input_info(
                        "error".to_string(),
                        TensorInfo::new(
                            //DataType::Float,
                            self.dtype,
                            Shape::from(None),
                            OutputName {
                                name: self.error.name()?,
                                index: 0,
                            },
                        ),
                    );
                    def.add_input_info(
                        "minimize".to_string(),
                        TensorInfo::new(
                            //DataType::Float,
                            self.dtype,
                            Shape::from(None),
                            OutputName {
                                name: self.minimize.name()?,
                                index: 0,
                            },
                        ),
                    );

                    def
                });
            let saved_model_saver = builder.inject(&mut self.scope)?;
            self.SavedModelSaver.replace(Some(saved_model_saver));
        }

        if !Path::new(name).exists() {
            fs::create_dir(name)?;
        }
        self.SavedModelSaver.borrow_mut().as_mut().unwrap().save(
            &self.session,
            &self.scope.graph(),
            &format!("{}/{}", name, self.name),
        )?;
        self.serialize_network(format!("{}/{}", name, self.name))?;

        Ok(())
    }

    /// load the saved model in the directory dir and restore it in self, removing the
    /// previous tensorflow graph and session.
    pub fn load(&mut self, dir: String) -> Result<(), Box<dyn Error>> {
        log::debug!("loading previously saved model..");
        let mut graph = Graph::new();
        //TODO: ensure we can access variables from graph or otherwise
        let bundle =
            SavedModelBundle::load(&self.session_options, &["serve", "train"], &mut graph, dir)?;
        let signature = bundle
            .meta_graph_def()
            .get_signature(REGRESS_METHOD_NAME)?
            .clone();
        self.session = bundle.session;
        self.Input =
            graph.operation_by_name_required(&signature.get_input("inputs")?.name().name)?;
        self.Label =
            graph.operation_by_name_required(&signature.get_input("label")?.name().name)?;
        self.error =
            graph.operation_by_name_required(&signature.get_input("error")?.name().name)?;
        self.minimize =
            graph.operation_by_name_required(&signature.get_input("minimize")?.name().name)?;

        Ok(())
    }

    ///feed in the given input and label tensors and run the session
    pub fn feed<'b, 'c: 'b, T>(
        mut run_args: SessionRunArgs<'b>,
        Input: &'c Operation,
        Label: &'c Operation,
        input_tensor: &'c Tensor<T>,
        label_tensor: &'c Tensor<T>,
        session: &Session,
    ) -> Result<(), Status>
    where
        T: tensorflow::TensorType + Clone + Send + Sync,
        <T as tensorflow::TensorType>::InnerType: Send + Sync,
    {
        Ok(())
    }
    // TODO: should also return error (need to structure output I'm allergic to tuples)
    // TODO: should pass in a threshold fitness to prevent over trainning
    /// Train the network with the given inputs and labels (must be synchronized in index order)
    ///
    ///**PARAMETERS**:
    ///
    /// * inputs: the inputs to the network as a collection of flattened 1D vector
    ///
    /// * labels: the labels for the inputs as a collection of flattened 1D vector
    ///
    pub fn train<T, I, const Ilen: usize, L, const Llen: usize>(
        &mut self,
        inputs: I,
        labels: L,
    ) -> Result<Vec<Tensor<T>>, Box<dyn Error>>
    where
        T: tensorflow::TensorType + Clone + Send + Sync,
        <T as tensorflow::TensorType>::InnerType: Send + Sync,
        I: IntoIterator<Item = [T; Ilen]> //also par_iter
            + IntoParallelIterator<Item = [T; Ilen]>,
        L: IntoIterator<Item = [T; Llen]> //also par_iter
            + IntoParallelIterator<Item = [T; Llen]>,
    {
        //TODO: do we lose variable state each session?
        //TODO: k-folding extension method
        log::debug!("trainning..");

        let mut input_tensors: Vec<Tensor<T>> = inputs
            .into_par_iter()
            .map(|input| {
                let res: Tensor<T> = Tensor::new(&[1u64, Ilen as u64])
                    .with_values(&input)
                    .unwrap()
                    .into();
                res
            })
            .collect::<Vec<Tensor<T>>>();
        let mut label_tensors: Vec<Tensor<T>> = labels
            .into_par_iter()
            .map(|label| {
                let res: Tensor<T> = Tensor::new(&[1u64, Llen as u64])
                    .with_values(&label)
                    .unwrap();
                res
            })
            .collect::<Vec<Tensor<T>>>();
        log::debug!("input_tensors.len(): {}", input_tensors.len());
        log::debug!("label_tensors.len(): {}", label_tensors.len());
        let batch_size = input_tensors.len();

        let res: Vec<SendableTensor<T>> = 
            //trainning_tensors
            input_tensors.into_iter().zip(label_tensors.into_iter()).par_bridge()
            .map(|(input_tensor, label_tensor)| {
                let input_tensor_rock: Tensor<T> = input_tensor.into();
                let label_tensor_rock: Tensor<T> = label_tensor.into();

                let mut run_args = SessionRunArgs::new();
                let error = run_args.request_fetch(&self.error, 0);
                let output = run_args.request_fetch(&self.Output_op, 0);
                //run_args.add_target(&self.minimize);
                run_args.add_target(&self.minimize);

                run_args.add_feed(&self.Input, 0, &input_tensor_rock);
                run_args.add_feed(&self.Label, 0, &label_tensor_rock);
                self.session.run(&mut run_args).unwrap();
                let error: SendableTensor<T> = run_args.fetch(error).unwrap().into();
                //print the error
                let tmp_error: Tensor<T> = error.clone().into();
                println!("error: {:?}", tmp_error);


                error
            })
            .collect::<Vec<SendableTensor<T>>>();

        Ok(res
            .into_iter()
            .map(|t| t.into())
            .collect::<Vec<Tensor<T>>>())
    }

    //TODO: k-fold a given batch of inputs given a ratio of validation to trainning (up to user to
    //      ensure validation isnt cross contaminated
    //TODO: search iterations should see different datasets/epochs of dataset (not actual epoch backprop) via k-folding
    //      also cross validate the k-fold

    ///Infer a given batch of inputs, returning the ordered outputs as a vector. This does not
    ///update the parameters or perform backpropagation.
    pub fn infer<'b, T, I, const Ilen: usize>(
        &mut self,
        inputs: I,
    ) -> Result<Vec<Tensor<T>>, Box<dyn Error>>
    where
        T: tensorflow::TensorType + Clone,
        <T as tensorflow::TensorType>::InnerType: Send + Sync,
        I: IntoIterator<Item = [T; Ilen]> + IntoParallelIterator<Item = [T; Ilen]>,
    {
        log::debug!("infering..");
        let mut result = vec![];
        let mut input_tensors: Vec<SendableTensor<T>> = inputs
            .into_par_iter()
            .map(|input| {
                let res: SendableTensor<T> = Tensor::new(&[1u64, Ilen as u64])
                    .with_values(&input)
                    .unwrap()
                    .into();
                res
            })
            .collect::<Vec<SendableTensor<T>>>();
        //cast sendable tensors into plain ol tensors
        let mut input_tensors: Vec<Tensor<T>> = input_tensors
            .into_iter()
            .map(|input| input.into())
            .collect::<Vec<Tensor<T>>>();

        log::debug!("input_tensors.len(): {}", input_tensors.len());
        for input_tensor in input_tensors.iter_mut() {
            let mut run_args = SessionRunArgs::new();
            let output = run_args.request_fetch(&self.Output_op, 0);
            run_args.add_feed(&self.Input, 0, &input_tensor);
            self.session.run(&mut run_args)?;
            let output: Tensor<T> = run_args.fetch(output)?;
            log::info!("input: {:?} output: {}", input_tensor, output);
            result.push(output);
        }
        Ok(result)
    }
    //TODO: function that returns the current parameters as an iterator of slices
}

//TODO: serialize any data outside of the graph, currently this isnt necessary and ideally we
//      deprecate this soon for GraphAPI.
///The serialized representation of the model outside of tensorflow graph.
#[derive(Serialize, Deserialize, Debug)]
struct SerializedNetwork {
    parent_search_name: String,
}
impl SerializedNetwork {
    fn new(parent_search_name: String) -> Self {
        Self { parent_search_name }
    }
    fn restore(self, net: &mut Brain) {}
}

#[cfg(test)]
mod tests {
    use crate::layers::*;
    use crate::*;
    #[test]
    fn test_initial() {
        let mut scope = Scope::new_root_scope();
        let opt = optimizer::GradientDescent()
            .learning_rate(0.01 as f32)
            .build(&mut scope)
            .unwrap();
        let mut Net = Brain()
            .name("test-initial")
            .num_inputs(2)
            //demonstrate internal method chaining builder alternative:
            //.add_layer(layers::std::new(20, activations::Relu()))
            .add_layer(Box::new(
                fully_connected::init()
                    .width(20)
                    .activation(activations::Relu()),
            ))
            .add_layer(fully_connected::layer(200, activations::Relu()))
            .add_layer(fully_connected::layer(200, activations::Relu()))
            .add_layer(fully_connected::layer(1, activations::Relu()))
            .dtype(DataType::Float)
            .optimizer(opt)
            .error(optimizer::l2())
            .build(&mut scope)
            .unwrap();

        //train the network
        let mut rrng = rand::thread_rng();
        // create 100 entries for inputs and outputs of xor
        for _ in 0..100 {
            let mut inputs = Vec::new();
            //let outputs: mut Vec<&[f32;1] = Vec::new();
            let mut outputs = Vec::new();
            for _ in 0..100 {
                // instead of the above, generate either 0 or 1 and cast to f32
                let input = [(rrng.gen::<u8>() & 1) as f32, (rrng.gen::<u8>() & 1) as f32];
                let output = [(input[0] as u8 ^ input[1] as u8) as f32];
                inputs.push(input);
                outputs.push(output);
            }
            let res = Net.train(inputs, outputs).unwrap();
            //println!("inputs: {:?} \n res: {:?}", inputs, res);
        }
        //save the model
        let uuid = Uuid::new_v4();
        let save_file = format!("test-initial-{}", uuid);
        Net.save(&save_file).unwrap();
        return;
    }
    #[test]
    fn test_gpu() {
        let mut scope = Scope::new_root_scope();
        let mut scope = scope.with_device("/gpu:0");
        let opt = optimizer::Adadelta()
            .learning_rate(half::f16::from_f32(0.01))
            .rho(half::f16::from_f32(0.95))
            .epsilon(half::f16::from_f32(1e-6))
            .build(&mut scope)
            .unwrap();
        //        let opt = optimizer::GradientDescent()
        //            .learning_rate(half::f16::from_f32(0.01))
        //            .dtype(DataType::Half)
        //            .build(&mut scope)
        //            .unwrap();
        let mut Net = Brain()
            .name("test-gpu")
            .num_inputs(2)
            //TODO: it would be nice if there was compile time warning if the target device ISA
            //      doesnt support the given type (also for tf_ops)
            .add_layer(fully_connected_zero_init::layer(2000, activations::Elu()))
            .add_layer(fully_connected_zero_init::layer(2000, activations::Elu()))
            .add_layer(fully_connected_zero_init::layer(2000, activations::Elu()))
            .add_layer(fully_connected_zero_init::layer(2000, activations::Elu()))
            .add_layer(fully_connected_zero_init::layer(2000, activations::Elu()))
            .add_layer(fully_connected_zero_init::layer(2000, activations::Elu()))
            .add_layer(fully_connected_zero_init::layer(1, activations::Tanh()))
            .dtype(DataType::Half)
            .optimizer(opt)
            .error(optimizer::l2())
            .build(&mut scope)
            .unwrap();

        //train the network
        let mut rrng = rand::thread_rng();
        // create 100 entries for inputs and outputs of xor
        //let mut counter: usize = 0;
        for _ in 0..25 {
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            let mut input_one = half::f16::from_f32(1.0);
            let mut input_two = half::f16::from_f32(0.0);
            let mut output_cast = half::f16::from_f32(1.0);
            for _ in 0..400 {
                //print the counter
                //counter += 1;
                //if counter % 1000 == 0 {
                //    println!("counter: {}", counter);
                //}
                // instead of the above, generate either 0 or 1 and cast to f32
                let input_one = half::f16::from_f32((rrng.gen::<u8>() & 1) as f32);
                let input_two = half::f16::from_f32((rrng.gen::<u8>() & 1) as f32);
                let input = [input_one, input_two];
                output_cast = half::f16::from_f32(
                    ((input[0].to_f32()) as u8 ^ input[1].to_f32() as u8) as f32,
                )
                .clone();
                let output = [output_cast];
                //cast input and outputs to bf16

                inputs.push(input);
                outputs.push(output);
            }
            assert_eq!(inputs.len(), outputs.len());
            let res = Net.train(inputs, outputs).unwrap();
        }
        //print the output of the network
        let mut input = vec![
            [half::f16::from_f32(0.0), half::f16::from_f32(1.0)],
            [half::f16::from_f32(1.0), half::f16::from_f32(0.0)],
            [half::f16::from_f32(0.0), half::f16::from_f32(0.0)],
            [half::f16::from_f32(1.0), half::f16::from_f32(1.0)],
        ];

        let output = Net.infer(input.clone()).unwrap();
        println!(
            "test gpu XOR test input: {:?} \n XOR test output: {:?}",
            input, output
        );

        //create a UUID string
        let uuid = Uuid::new_v4().to_string();
        let save_file = format!("test-builder-{}", uuid);
        //save the model
        Net.save(save_file.as_str()).unwrap();

        return;
    }
    #[test]
    fn test_mixed_precision() {
        let mut scope = Scope::new_root_scope();
                let opt = optimizer::GradientDescent()
                    .learning_rate(tensorflow::BFloat16::from(0.01 as f32))
                    .build(&mut scope)
                    .unwrap();
//        let opt = optimizer::Adadelta()
//            .learning_rate(tensorflow::BFloat16::from(0.01 as f32))
//            .rho(tensorflow::BFloat16::from(0.95 as f32))
//            .epsilon(tensorflow::BFloat16::from(1e-6 as f32))
//            .build(&mut scope)
//            .unwrap();

        let mut Net = Brain()
            .name("test-mixed_precision")
            .num_inputs(2)
            .add_layer(fully_connected::layer(200, activations::Elu()))
            .add_layer(fully_connected::layer(200, activations::Elu()))
            .add_layer(fully_connected::layer(200, activations::Elu()))
            .add_layer(fully_connected::layer(200, activations::Elu()))
            .add_layer(fully_connected::layer(1, activations::Tanh()))
            .dtype(DataType::BFloat16)
            .optimizer(opt)
            .error(optimizer::l2())
            .build(&mut scope)
            .unwrap();
        let mut rrng = rand::thread_rng();
        // create 100 entries for inputs and outputs of xor

        for _ in 0..250 {
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            //            let mut input_one = half::bf16::from_f32(1.0);
            //            let mut input_two = half::bf16::from_f32(0.0);
            //            let mut output_cast = half::bf16::from_f32(1.0);
            let mut input_one = tensorflow::BFloat16::from(1.0 as f32);
            let mut input_two = tensorflow::BFloat16::from(0.0 as f32);
            let mut output_cast = tensorflow::BFloat16::from(1.0 as f32);
            for _ in 0..100 {
                // instead of the above, generate either 0 or 1 and cast to f32
                //let input_one = half::bf16::from_f32((rrng.gen::<u8>() & 1) as f32);
                //let input_two = half::bf16::from_f32((rrng.gen::<u8>() & 1) as f32);
                let input_one = tensorflow::BFloat16::from((rrng.gen::<u8>() & 1) as f32);
                let input_two = tensorflow::BFloat16::from((rrng.gen::<u8>() & 1) as f32);
                let input = [input_one, input_two];
                //output_cast = half::bf16::from_f32(
                //    ((input[0].to_f32()) as u8 ^ input[1].to_f32() as u8) as f32,
                //)
                //.clone();
                output_cast = tensorflow::BFloat16::from(
                    (f32::from(input[0]) as u8 ^ f32::from(input[1]) as u8) as f32,
                );
                let output = [output_cast];
                //cast input and outputs to bbf16

                inputs.push(input);
                outputs.push(output);
            }
            assert_eq!(inputs.len(), outputs.len());
            let res = Net.train(inputs, outputs).unwrap();
        }
        //print the output of the network
        //        let mut input = vec![
        //            [half::bf16::from_f32(0.0), half::bf16::from_f32(1.0)],
        //            [half::bf16::from_f32(1.0), half::bf16::from_f32(0.0)],
        //            [half::bf16::from_f32(0.0), half::bf16::from_f32(0.0)],
        //            [half::bf16::from_f32(1.0), half::bf16::from_f32(1.0)],
        //        ];
        let mut input = vec![
            [
                tensorflow::BFloat16::from(0.0 as f32),
                tensorflow::BFloat16::from(1.0 as f32),
            ],
            [
                tensorflow::BFloat16::from(1.0 as f32),
                tensorflow::BFloat16::from(0.0 as f32),
            ],
            [
                tensorflow::BFloat16::from(0.0 as f32),
                tensorflow::BFloat16::from(0.0 as f32),
            ],
            [
                tensorflow::BFloat16::from(1.0 as f32),
                tensorflow::BFloat16::from(1.0 as f32),
            ],
        ];

        let output = Net.infer(input.clone()).unwrap();
        //cast output to f32
        let output = output.iter().map(|x| (x[0]).into()).collect::<Vec<f32>>();
        println!(
            "mixed precision XOR test input: {:?} \n XOR test output: {:?}",
            input, output
        );
    }
    #[test]
    pub fn test_norm_net() {
        let mut scope = Scope::new_root_scope();
        let opt = optimizer::GradientDescent()
            .learning_rate(half::f16::from_f32(0.01 as f32))
            .build(&mut scope)
            .unwrap();
        let mut Net = Brain()
            .name("test-norm-net")
            .num_inputs(2)
            .add_layer(norm::layer(20, activations::Tanh()))
            .add_layer(norm::layer(20, activations::Tanh()))
            .add_layer(norm::layer(20, activations::Tanh()))
            .add_layer(norm::layer(1, activations::Tanh()))
            .dtype(DataType::Half)
            .optimizer(opt)
            .error(optimizer::l1())
            .build(&mut scope)
            .unwrap();
        let mut rrng = rand::thread_rng();
        // create 100 entries for inputs and outputs of xor
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for _ in 0..20{
            for _ in 0..200 {
                // instead of the above, generate either 0 or 1 and cast to f32
                //let input_one = half::bf16::from_f32((rrng.gen::<u8>() & 1) as f32);
                //let input_two = half::bf16::from_f32((rrng.gen::<u8>() & 1) as f32);
                let input_one = half::f16::from_f32((rrng.gen::<u8>() & 1) as f32);
                let input_two = half::f16::from_f32((rrng.gen::<u8>() & 1) as f32);
                let input = [input_one, input_two];
                //output_cast = half::bf16::from_f32(
                //    ((input[0].to_f32()) as u8 ^ input[1].to_f32() as u8) as f32,
                //)
                //.clone();
                let output_cast = half::f16::from_f32(
                    (f32::from(input[0]) as u8 ^ f32::from(input[1]) as u8) as f32,
                );
                let output = [output_cast];
                //cast input and outputs to bbf16

                inputs.push(input.clone());
                outputs.push(output.clone());
            }
            let error = Net.train(inputs.clone(), outputs.clone()).unwrap();  
            inputs.clear();
            outputs.clear();
            //print the last 10 errors with newlines
            error.iter().rev().take(10).for_each(|x| println!("{:?}", x));
        }
        assert_eq!(inputs.len(), outputs.len());
        //print the output of the network
        let mut input = vec![
            [half::f16::from_f32(0.0), half::f16::from_f32(1.0)],
            [half::f16::from_f32(1.0), half::f16::from_f32(0.0)],
            [half::f16::from_f32(0.0), half::f16::from_f32(0.0)],
            [half::f16::from_f32(1.0), half::f16::from_f32(1.0)],
        ];
        let res = Net.infer(input.clone()).unwrap();
        println!("input: {:?} \n output: {:?}", input, res);
    }
}
