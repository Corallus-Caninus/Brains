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
use anyhow::Context;
use half::bf16;
use half::f16;
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
use rayon::iter::ParallelBridge;
use rayon::prelude::*;
use std::collections::HashMap;
use std::os;
use std::rc::Rc;
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
//function types
//@DEPRECATED, we are removing function types in favor of trait bounds
//use layers::Layer;
use layers::*;
//TODO: @DEPRECATED
//use activations::Activation;
//pub mod activations;
//pub mod layers;
//use activations::*;
//use anyhow::anyhow;
//#[macro_use]
//extern crate derive_builder;
extern crate tensorflow;

//      consider using a builder struct that is a master class for new constructor, some of
//      this isnt initialized, only need a builder for user parameters
//builder should also take functions for layers and activations as a functional struct, possibly
//reuse architecture for this but rename and only struct ActivatedLayer

//TODO: builder takes trait bounds on ActivatedLayer as a form generically returning self like iter()
//TODO: why isnt Layer a trait object here? will this conflict with heterogeneous layer
//      architectures at compile time?
pub struct BrainBuilder<Layer: BuildLayer + LayerAccessor + ConfigurableLayer> {
    name: String,
    num_inputs: u64,
    layers: Vec<Layer>,
    learning_rate: f32,
    error_power: f32,
}
//TODO: This causes a temporary value dropped while building
///Constructor to initialize the BrainBuilder
pub fn Brain<Layer>() -> BrainBuilder<Layer>
where
    Layer: BuildLayer + LayerAccessor + ConfigurableLayer,
{
    BrainBuilder {
        name: "Brain".to_string(),
        num_inputs: 0,
        layers: Vec::new(),
        learning_rate: 0.01,
        error_power: 1.0,
    }
}
//TODO: a trait or some way to do std_layer instead of add_layer(std_layer)
impl<Layer> BrainBuilder<Layer>
where
    Layer: BuildLayer + LayerAccessor + ConfigurableLayer,
{
    pub fn add_layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }
    pub fn name(mut self, name: String) -> Self {
        self.name = name;
        self
    }
    pub fn num_inputs(mut self, num_inputs: u64) -> Self {
        self.num_inputs = num_inputs;
        self
    }
    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }
    pub fn error_power(mut self, error_power: f32) -> Self {
        self.error_power = error_power;
        self
    }
    pub fn build(mut self) -> Result<Brain, Status> {
        //TODO: ~Just Rust Things~
        let mut layers = Vec::new();
        for i in 0..self.layers.len() {
            layers.push(self.layers.pop().unwrap());
        }

        //move self.layers to a local variable named layers since self.layers isnt clone
        Brain::new(
            self.name,
            layers,
            self.num_inputs,
            self.learning_rate,
            self.error_power,
        )
    }
}

pub struct Brain {
    /// Tensorflow objects for user abstraction from Tensorflow
    scope: Scope,
    session: Session,
    session_options: SessionOptions,
    ///each layers output for the network
    net_layers: Vec<Output>,
    ///all the trainable parameters of the network
    net_vars: Vec<Variable>,
    ///Operation hooks to interact with the graph (kept here for serialization)
    Input: Operation,
    Label: Operation,
    Output_op: Operation,
    Error: Operation,
    ///variables to be minimized
    minimize_vars: Vec<Variable>,
    ///regression operation
    minimize: Operation,
    ///class for serializing, saving and loading the model
    SavedModelSaver: RefCell<Option<SavedModelSaver>>,
    ///user specified name of this model
    name: String,
    ///the error power for scaling the error gradient's pressure on the weights
    error_power: f32,
}
impl Brain {
    //TODO: type safety: use trait bounds to allow for using bigints etc for counting//indexing
    //      types.
    pub fn new<Layer: BuildLayer + LayerAccessor + ConfigurableLayer>(
        name: String,
        //TODO: a vec of layers here will suffice for now, but this will be a builder pattern as
        //soon as possible
        layers: Vec<Layer>,
        num_inputs: u64,
        //activation: Activation,
        learning_rate: f32,
        error_power: f32,
    ) -> Result<Brain, Status> {
        let mut scope = Scope::new_root_scope();

        let mut input_size = num_inputs;
        let mut output_size = layers[0].get_width();
        //TODO: may want more than a vector rank dimension
        let Input = ops::Placeholder::new()
            .dtype(DataType::Float)
            .shape([1u64, input_size])
            .build(&mut scope.with_op_name("input"))?;
        let Label = ops::Placeholder::new()
            .dtype(DataType::Float)
            .shape([1u64, layers[layers.len() - 1].get_width()])
            .build(&mut scope.with_op_name("label"))?;

        //CONSTRUCT NETWORK
        //TODO: extract this to a builder pattern for constructor readability.
        //TODO: builder should also take a function that defines this, really we need the input and
        //output to get all the checkpointing/trainning features. helper functions can make this
        //function easier to define elsewhere

        //trainnable variables
        let mut net_vars = vec![];
        //each layer as a TF Operation
        let mut net_layers = vec![];

        let mut layer_input_iter = Input.clone();
        for mut layer in layers {
            //TODO: assign Input to each layer. extract this to builder later
            //TODO: this should already be done in layer builder
            //      (not the aformbentioned Brain builder)
            //TODO: need to setup the Input hook in the builder.
            //layer.get_input()?(layer_input_iter.clone().into());

            layer.input(layer_input_iter.clone());
            layer.input_size(input_size);
            input_size = layer.get_width();
            output_size = layer.get_width();
            layer.output_size(output_size);
            //TODO: end of extraction routine

            let parameters = layer.build_layer(&mut scope)?;
            let vars = parameters.variables;
            let output = parameters.output;

            net_vars.extend(vars);
            net_layers.push(output.clone());

            layer_input_iter = parameters.operation;
        }
        let Output = net_layers.last().unwrap().clone();
        let Output_op = Output.operation.clone();

        let options = SessionOptions::new();
        let SavedModelSaver = RefCell::new(None);

        //TODO: pass this in? this should be modular so new implementations can be tried
        let mut optimizer = AdadeltaOptimizer::new();
        optimizer.set_learning_rate(ops::constant(learning_rate, &mut scope)?);
        // let mut optimizer =
        //     GradientDescentOptimizer::new(ops::constant(learning_rate, &mut scope)?);

        // DEFINE ERROR FUNCTION //
        //TODO: pass this in conditionally, give user output and label with
        //      a partial constructor then they supply error and construction is complete
        //      two structs? whats standard functionally for partial construction?
        //  ^this is important for reinforcement learning where label difference should be output gradient direction (multiplication)
        //TODO: use setters instead and have a default object behaviour.

        //default error is pythagorean distance
        let Error = ops::sqrt(
            ops::pow(
                ops::sub(Output.clone(), Label.clone(), &mut scope)?,
                ops::constant(2.0 as f32, &mut scope)?,
                &mut scope,
            )?,
            &mut scope,
        )?;
        let Error = ops::pow(
            Error.clone(),
            ops::constant(error_power, &mut scope).unwrap(),
            &mut scope.with_op_name("error"),
        )?;

        let (minimize_vars, minimize) = optimizer
            .minimize(
                &mut scope,
                Error.clone().into(),
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
            session: session,
            session_options: options,
            net_layers,
            net_vars,
            Input,
            Label,
            Output_op,
            Error,
            error_power,
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
                            DataType::Float,
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
                            DataType::Float,
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
                            DataType::Float,
                            Shape::from(None),
                            OutputName {
                                name: self.Error.name()?,
                                index: 0,
                            },
                        ),
                    );
                    def.add_input_info(
                        "minimize".to_string(),
                        TensorInfo::new(
                            DataType::Float,
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
        self.Error =
            graph.operation_by_name_required(&signature.get_input("error")?.name().name)?;
        self.minimize =
            graph.operation_by_name_required(&signature.get_input("minimize")?.name().name)?;

        Ok(())
    }

    /// Train the network with the given inputs and labels (must be synchronized in index order)
    ///
    ///**PARAMETERS**:
    ///
    /// * inputs: the inputs to the network as a flattened 1D vector
    ///
    /// * labels: the labels for the inputs as a flattened 1D vector
    ///
    /// NOTE: labels and inputs must align with the network input and output vector.
    pub fn train<T: tensorflow::TensorType>(
        // &mut self,
        &mut self,
        inputs: Vec<Vec<T>>,
        labels: Vec<Vec<T>>,
    ) -> Result<Vec<Tensor<f32>>, Box<dyn Error>> {
        //TODO: k-folding extension method
        assert_eq!(inputs.len(), labels.len());
        log::debug!("trainning..");
        let mut result = vec![];

        let mut input_tensor: Tensor<T> = Tensor::new(&[1u64, inputs[0].len() as u64]);
        let mut label_tensor: Tensor<T> = Tensor::new(&[1u64, labels[0].len() as u64]);

        log::debug!("inputs.len(): {}", inputs.len());
        log::debug!("{}", inputs[0].len());
        log::debug!("{}", labels[0].len());

        let mut input_iter = inputs.into_iter();
        let mut label_iter = labels.into_iter();

        //initialize iteration
        input_iter.next();
        label_iter.next();
        let mut i = 0;
        let mut avg_t = vec![];
        //TODO: dont loop this send it all to the VRAM and unroll the n-1 dimensional input tensor,
        loop {
            // start a timer
            let start = Instant::now();

            i += 1;
            let input = input_iter.next();
            let label = label_iter.next();
            if input.is_none() || label.is_none() {
                break;
            }
            let input = input.unwrap();
            let label = label.unwrap();
            // now get input and label as slices
            let input = input.as_slice();
            let label = label.as_slice();
            // now assign the input and label to the tensor
            for i in 0..input.len() {
                input_tensor[i] = input[i].clone();
            }
            for i in 0..label.len() {
                label_tensor[i] = label[i].clone();
            }

            let mut run_args = SessionRunArgs::new();
            run_args.add_target(&self.minimize);

            let error_squared_fetch = run_args.request_fetch(&self.Error, 0);
            let output = run_args.request_fetch(&self.Output_op, 0);
            run_args.add_feed(&self.Input, 0, &input_tensor);
            run_args.add_feed(&self.Label, 0, &label_tensor);
            self.session.run(&mut run_args)?;

            let res: Tensor<f32> = run_args.fetch(error_squared_fetch)?;
            let output: Tensor<T> = run_args.fetch(output)?;

            // get how long has passed
            let elapsed = start.elapsed();
            avg_t.push(elapsed.as_secs_f32());

            // update the moving average for time
            let average = avg_t.iter().sum::<f32>() / avg_t.len() as f32;

            log::info!(
                "training on {}\n input: {:?} label: {:?} error: {} output: {} seconds/epoch: {:?}",
                i,
                input,
                label,
                res,
                output,
                average
            );

            result.push(res);
        }

        Ok(result)
    }

    //TODO: k-fold a given batch of inputs given a ratio of validation to trainning (up to user to
    //      ensure validation isnt cross contaminated
    //TODO: search iterations should see different datasets/epochs of dataset (not actual epoch backprop) via k-folding
    //      also cross validate the k-fold

    /// forward pass and return the output of the network.
    pub fn infer<T: tensorflow::TensorType>(
        &self,
        inputs: Tensor<T>,
    ) -> Result<Tensor<T>, Box<dyn Error>> {
        let mut run_args = SessionRunArgs::new();
        let output = run_args.request_fetch(&self.Output_op, 0);
        run_args.add_feed(&self.Input, 0, &inputs);

        self.session.run(&mut run_args)?;

        let output: Tensor<T> = run_args.fetch(output)?;
        Ok(output)
    }

    /// Online reinforcement learning method.
    ///
    /// Takes the given inputs and fitness function and backprops the network once,
    /// returning the output output vector.
    /// This should be called as an online (realtime) reinforcment learning technique
    /// where labels can be formed given a fitness function.
    ///
    ///PARAMETERS:
    /// returns the output from the network for online learning implementation.
    ///
    /// Fitness function takes a 1D tensor of length output and returns a label tensor of same shape.
    pub fn evaluate(
        &mut self,
        inputs: Vec<f32>,
        //TODO: make this internal to class as builder so we dont clone the dyn ref alot ITL
        //TODO: refactor the names here to make this more intuitive
        fitness_function: Box<dyn Fn(&Tensor<f32>) -> Tensor<f32>>,
    ) -> Result<Tensor<f32>, Box<dyn Error>> {
        // let mut label_tensor: Tensor<T> = Tensor::new(&[1u64, labels.len() as u64]);
        // now assign the input and label to the tensor
        let mut input_tensor = Tensor::new(&[1u64, inputs.len() as u64]);
        for i in 0..inputs.len() {
            input_tensor[i] = inputs[i].clone();
        }

        // call infer to get the output
        let outputs = self.infer(input_tensor.clone())?;

        //create labels using fitness function
        let reward = fitness_function(&outputs);

        //backprop
        let mut run_args = SessionRunArgs::new();
        run_args.add_target(&self.minimize);

        let error_squared_fetch = run_args.request_fetch(&self.Error, 0);
        // set output feed manually
        //TODO: runtime says we need to feed input again so create a placeholder tensor of 0
        run_args.add_feed(&self.Input, 0, &input_tensor);
        // run_args.add_feed(&self.Output_op, 0, &outputs);
        run_args.add_feed(&self.Label, 0, &reward);
        self.session.run(&mut run_args)?;

        let cur_error: Tensor<f32> = run_args.fetch(error_squared_fetch)?;

        //return the output
        Ok(outputs.clone())
    }
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
    use crate::*;
    //TODO: these need a better constructor at the brain level, fix after layer is refactored
    #[test]
    fn test_initial() {
        //first, we create the network as a Vector of Layers
        //TODO: internal builder with method chaining instead of vec of Layers
        //TODO: can build a call stack with vec push fn's then call
        //      sequentially with build for lazy builder.
        //      Since each argument is len 1 use a tuple Vec<(fn, arg)>
        //      use fn -> arg repetition pattern in builder trait.
        //      a lazy builder on a lazy builder.
        //let network = std().width(10).activation(activations::tanh(100)).std().width(10).activation(activations::tanh(100));
        //TODO: extract this routine into builder
        let network = vec![
            //layers::std_layer::new(2, activations::Tanh(10)),
            layers::std_layer::new(100, activations::Tanh(10)),
            layers::std_layer::new(100, activations::Tanh(10)),
            layers::std_layer::new(1, activations::Sigmoid(100)),
        ];

        let mut Net = Brain::new("test-net".to_string(), network, 2, 32.0, 10.0).unwrap();

        //train the network
        let mut rrng = rand::thread_rng();
        // create 100 entries for inputs and outputs of xor
        for _ in 0..100 {
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            for _ in 0..100 {
                // instead of the above, generate either 0 or 1 and cast to f32
                let input = vec![(rrng.gen::<u8>() & 1) as f32, (rrng.gen::<u8>() & 1) as f32];
                let output = vec![(input[0] as u8 ^ input[1] as u8) as f32];
                inputs.push(input);
                outputs.push(output);
            }
            Net.train(inputs, outputs).unwrap();
            println!("trained a batch");
        }
        //save the model
        let uuid = Uuid::new_v4();
        let save_file = format!("test-initial-{}", uuid);
        Net.save(&save_file).unwrap();
        return;
    }
    //TODO:
    #[test]
    fn test_builder() {
        //TODO: Brain().inputs(2).layer(layers::std_layer::new(1000, activations::Tanh(10))).layer(layers::std_layer::new(1, activations::Tanh(10))).build();
        //TODO: also expose optional configuration such as Brain().inputs(2).dtype(bf16).layers::std_layer::new(1000, activations::Tanh(10)).build();

        //let mut Net = Brain::new("test-net", network, 2, 32.0, 10.0).unwrap();
        let mut Net = Brain()
            .num_inputs(2)
            .add_layer(layers::std_layer::new(100, activations::Tanh(10)))
            .add_layer(layers::std_layer::new(100, activations::Tanh(10)))
            .add_layer(layers::std_layer::new(1, activations::Sigmoid(100)))
            .build()
            .unwrap();

        //train the network
        let mut rrng = rand::thread_rng();
        // create 100 entries for inputs and outputs of xor
        for _ in 0..100 {
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            for _ in 0..100 {
                // instead of the above, generate either 0 or 1 and cast to f32
                let input = vec![(rrng.gen::<u8>() & 1) as f32, (rrng.gen::<u8>() & 1) as f32];
                let output = vec![(input[0] as u8 ^ input[1] as u8) as f32];
                inputs.push(input);
                outputs.push(output);
            }
            Net.train(inputs, outputs).unwrap();
            println!("trained a batch");
        }
        //create a UUID string
        let uuid = Uuid::new_v4().to_string();
        let save_file = format!("test-builder-{}", uuid);
        //save the model
        Net.save(save_file.as_str()).unwrap();

        return;
    }
}
//TODO: restore the below unittests with builder once the above test passes
//    fn test_net() {
//        log::debug!("test_net");
//        //call the main function
//        use crate::*;
//
//        //CONSTRUCTION//
//        let mut hidden_layer = std_layer().input_size(10).output_size(10).activation(tanh(10));
//        let layers = vec![
//            //TODO: prefer a function constructor like Keras and just size of this layer, also stateful
//            //      builder should track prev and next sizes
//            //e.g.:
//            //let mynet = Brain().std_layer(5, relu).std_layer(10, relu).std_layer(5, relu),
//            hidden_layer,
//            hidden_layer,
//            hidden_layer,
//            //std_layer().input_size(10).output_size(10).activation(relu(10)),
//            //std_layer().input_size(10).output_size(10).activation(relu(10)),
//            //std_layer().input_size(10).output_size(10).activation(relu(10)),
//            //std_layer().input_size(10).output_size(10).activation(relu(10)),
//        ];
//        //let mut norm_net = Brain::new("test_net",2, 1, 10, 10, layers::std_layer(), activations::tanh(10), 1.0, 5 as f32).unwrap();
//        let mut norm_net = Brain::new("test_net", layers, 1.0, 5 as f32).unwrap();
//
//        //FITNESS FUNCTION//
//        //TODO: auto gen labels from outputs and fitness function.
//
//        //TRAIN//
//        let mut rrng = rand::thread_rng();
//        let mut inputs = Vec::new();
//        let mut outputs = Vec::new();
//        // create 100 entries for inputs and outputs of xor
//        for _ in 0..1000 {
//            // instead of the above, generate either 0 or 1 and cast to f32
//            let input = vec![(rrng.gen::<u8>() & 1) as f32, (rrng.gen::<u8>() & 1) as f32];
//            let output = vec![(input[0] as u8 ^ input[1] as u8) as f32];
//
//            inputs.push(input);
//            outputs.push(output);
//        }
//
//        norm_net.train(inputs, outputs).unwrap();
//    }
//
//    #[test]
//    fn test_serialization() {
//        log::debug!("test_serialization");
//        //call the main function
//        use crate::*;
//
//        //CONSTRUCTION//
//        let mut norm_net = Brain::new("test_serialization",2, 1, 20, 15, layers::std_layer(), activations::tanh(10), 0.01, 5 as f32).unwrap();
//        //TRAIN//
//        let mut rrng = rand::thread_rng();
//        let mut inputs = Vec::new();
//        let mut outputs = Vec::new();
//        // create 100 entries for inputs and outputs of xor
//        for _ in 0..10 {
//            // instead of the above, generate either 0 or 1 and cast to f32
//            let input = vec![(rrng.gen::<u8>() & 1) as f32, (rrng.gen::<u8>() & 1) as f32];
//            let output = vec![(input[0] as u8 ^ input[1] as u8) as f32];
//
//            inputs.push(input);
//            outputs.push(output);
//        }
//
//        norm_net.train(inputs.clone(), outputs.clone()).unwrap();
//
//        // save the network
//        norm_net
//            .save()
//            .unwrap();
//
//        //load the network
//        let mut path = "".to_string();
//        //NOTE: dont ever call a string something else in your crates or someone I know will find you.
//        let _ = std::path::PathBuf::new();
//        for entry in fs::read_dir("test_serialization/").unwrap() {
//            let entry = entry.unwrap();
//            let is_dir = entry.path();
//            if !is_dir.is_dir() {
//                continue;
//            } else {
//                path = is_dir.clone().to_str().unwrap().to_string();
//            }
//            log::debug!("{:?}", path);
//        }
//        let path = path.to_string();
//        log::debug!("{:?}", path);
//
//        norm_net.load(path.to_string()).unwrap();
//
//        norm_net.train(inputs, outputs).unwrap();
//    }
//
//    #[test]
//    fn test_checkpoint(){
//        log::debug!("test_checkpoint");
//        use crate::*;
//        //CONSTRUCTION//
//        let mut norm_net = Brain::new("test_checkpoint",2, 1, 200, 96, layers::std_layer(), activations::tanh(10), 10.0, 5 as f32).unwrap();
//        //TRAIN//
//        let mut rrng = rand::thread_rng();
//        // create entries for inputs and outputs of xor
//        //TODO: window size and trag_iterations is hyperparameter for arch search. they should exist in shared struct or function parameter
//        //TODO: how can we train this in RL? need to store window and selection_pressure in class state
//        //TODO: this needs to happen on initialization
//        for _ in 0..15{
//            let mut inputs = Vec::new();
//            let mut outputs = Vec::new();
//            for _ in 0..10 {
//                // instead of the above, generate either 0 or 1 and cast to f32
//                let input = vec![(rrng.gen::<u8>() & 1) as f32, (rrng.gen::<u8>() & 1) as f32];
//                let output = vec![(input[0] as u8 ^ input[1] as u8) as f32];
//
//                inputs.push(input);
//                outputs.push(output);
//            }
//            // TEST TRAIN
//            norm_net.train_checkpoint_search(inputs.clone(), outputs.clone(),  2).unwrap();
//
//            // TEST LOAD
//            norm_net.load_checkpoint_search(0.001).unwrap();
//        }
//    }
//    #[test]
//    fn test_infer(){
//        log::debug!("test_inference");
//        use crate::*;
//        //CONSTRUCTION//
//        let mut norm_net = Brain::new("test_inference",2, 1, 200, 96, layers::std_layer(), activations::tanh(10), 10.0, 5 as f32).unwrap();
//        //TRAIN//
//        let mut rrng = rand::thread_rng();
//        // create entries for inputs and outputs of xor
//        for _ in 0..10{
//            let mut inputs:Tensor<f32> = Tensor::new(&[1u64, 2 as u64]);
//            let res: Tensor<f32>= norm_net.infer(inputs).unwrap();
//        }
//    }
//    #[test]
//    fn test_evaluate(){
//        log::debug!("test_evaluate");
//        use crate::*;
//        //CONSTRUCTION//
//        let layers = vec![layers::std_layer(), layers::std_layer()];
//        let mut norm_net = Brain::new("test_evaluate",2, 1, 200, 96, layers::std_layer(), activations::tanh(10), 1.0, 10 as f32).unwrap();
//        //TRAIN//
//        let mut rrng = rand::thread_rng();
//
//        //TODO: this doesnt make much sense since we arent implementing state-action-reward tables
//        let fitness_function = Box::new(|outputs: &Tensor<f32>| -> Tensor<f32> {
//            //TODO: which scalars in the output vector do we want to maximize? a good default fitness function
//            //      is a scalar constant to each entry in the vector or power term.
//            //pass a function with a derivative that goes to zero or 1?
//            let mut res_tensor = Tensor::new(outputs.dims());
//            res_tensor[0] = outputs[0]/outputs[0];
//            res_tensor
//        });
//        // create entries for inputs and outputs of xor
//        for _ in 0..30{
//            // same as above but push a vec with two random floats
//            let inputs = vec![(rrng.gen::<f32>()), (rrng.gen::<f32>())];
//            let res: Tensor<f32> =  norm_net.evaluate(inputs, 1, fitness_function.clone()).unwrap();
//        }
//
//        //SEARCH RECOVERED MODEL:
//        norm_net.load_checkpoint_search(0.001).unwrap();
//        for _ in 0..10{
//            let inputs = vec![(rrng.gen::<f32>()), (rrng.gen::<f32>())];
//            let res: Tensor<f32> =  norm_net.evaluate(inputs, 1, fitness_function.clone()).unwrap();
//        }
//    }
//    //TODO: test layer modularity
//    // #[test]
//    // fn test_architectures(){}
//}
