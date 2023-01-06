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

pub struct BrainBuilder<'a, Layer: BuildLayer + LayerAccessor + ConfigurableLayer> {
    name: &'a str,
    num_inputs: u64,
    input_type: DataType,
    //TODO: this is only a trait obj because optimizer isnt clone for method chaining
    Optimizer: Option<Box<dyn Optimizer>>,
    layers: Vec<Layer>,
}
//TODO: This causes a temporary value dropped while building
///Constructor to initialize the BrainBuilder
pub fn Brain<'a, Layer>() -> BrainBuilder<'a, Layer>
where
    Layer: BuildLayer + LayerAccessor + ConfigurableLayer,
{
    BrainBuilder {
        name: "Brain",
        num_inputs: 0,
        input_type: DataType::Float,
        Optimizer: None,
        layers: Vec::new(),
    }
}
//TODO: a trait or some way to do std_layer instead of add_layer(std_layer)
impl<'a, Layer> BrainBuilder<'a, Layer>
where
    Layer: BuildLayer + LayerAccessor + ConfigurableLayer,
    //Opt: Optimizer,
{
    pub fn add_layer(mut self, layer: Layer) -> Self {
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
    pub fn input_type(mut self, input_type: DataType) -> Self {
        self.input_type = input_type;
        self
    }
    pub fn optimizer(mut self, optimizer: Box<dyn Optimizer>) -> Self {
        self.Optimizer = Some(optimizer);
        self
    }
    pub fn build(mut self, scope: &mut Scope) -> Result<Brain, Status> {
        //TODO: ~Just Rust Things~
        let mut layers = Vec::new();
        for i in 0..self.layers.len() {
            layers.push(self.layers.pop().unwrap());
        }
        layers.reverse();

        //TODO: default condition for optimizer is verbose because tensorflow-rs
        //doesnt use a trait for constructing optimizers (yet).
        if self.Optimizer.is_none() {
            let sgd = GradientDescentOptimizer::new(ops::constant(0.01 as f32, scope)?);
            return Brain::new(
                self.name.to_string(),
                layers,
                Some(Box::new(sgd)),
                self.num_inputs,
                self.input_type,
                scope,
            );
        } else {
            return Brain::new(
                self.name.to_string(),
                layers,
                self.Optimizer,
                self.num_inputs,
                self.input_type,
                scope,
            );
        }
    }
}

pub struct Brain<'a> {
    /// Tensorflow objects for user abstraction from Tensorflow
    scope: &'a mut Scope,
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
}
impl<'a> Brain<'a> {
    //TODO: type safety: use trait bounds to allow for using bigints etc for counting//indexing
    //      types.
    pub fn new<Layer: BuildLayer + LayerAccessor + ConfigurableLayer>(
        name: String,
        //TODO: a vec of layers here will suffice for now, but this will be a builder pattern as
        //soon as possible
        layers: Vec<Layer>,
        //TODO: optimizer,
        optimizer: Option<Box<dyn Optimizer>>,
        num_inputs: u64,
        input_type: DataType,
        scope: &mut Scope,
    ) -> Result<Brain, Status> {
        //propagate an error if optimizer is none
        let optimizer = optimizer.unwrap();

        let mut scope = scope;

        let mut input_size = num_inputs;
        let mut output_size = layers[0].get_width();
        //TODO: may want more than a vector rank dimension, use this with an axis
        //unroll to batch in inputs therefor: +1 Rank
        let Input = ops::Placeholder::new()
            //.dtype(DataType::Float)
            .dtype(input_type)
            .shape([1u64, input_size])
            .build(&mut scope.with_op_name("input"))?;
        let Label = ops::Placeholder::new()
            //.dtype(DataType::Float)
            .dtype(layers.last().unwrap().get_dtype())
            .shape([1u64, *layers[layers.len() - 1].get_width()])
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

            layer = layer.input(layer_input_iter.clone());
            layer = layer.input_size(input_size);
            //TODO: refactor this
            input_size = *layer.get_width();
            output_size = &input_size;
            layer = layer.output_size(*output_size);
            //TODO: end of extraction routine

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

        // let mut optimizer =
        //     GradientDescentOptimizer::new(ops::constant(learning_rate, &mut scope)?);

        // DEFINE ERROR FUNCTION //
        // TODO: pass this in and offer some default helper functions within the framework
        //default error is distance, we should expose this for cross entropy etc
        let Error = ops::square(
            ops::abs(
                ops::sub(
                    Output.clone(),
                    Label.clone(),
                    &mut scope.with_op_name("error"),
                )?,
                &mut scope,
            )?,
            &mut scope,
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

    // TODO: should also return error (need to structure output I'm allergic to tuples)
    // TODO: should pass in a threshold fitness to prevent over trainning
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
        inputs: &Vec<Vec<T>>,
        labels: &Vec<Vec<T>>,
    ) -> Result<Vec<Tensor<f32>>, Box<dyn Error>> {
        //TODO: do we lose variable state each session?
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

        let mut i = 0;
        let mut avg_t = vec![];
        //TODO: dont loop this send it all to the VRAM and unroll the n-1 dimensional input tensor,
        //its up to the architect to ensure this fits in memory
        //unroll op occurs in graph declaration (Brain constructor).
        //this just makes a +1 Rank
        //tensor from Vec<Vec<T>>, runs session and creates a Vec<Vec<T>> from the output
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

            let error = run_args.request_fetch(&self.Error, 0);
            let output = run_args.request_fetch(&self.Output_op, 0);
            run_args.add_feed(&self.Input, 0, &input_tensor);
            run_args.add_feed(&self.Label, 0, &label_tensor);
            self.session.run(&mut run_args)?;

            let res: Tensor<f32> = run_args.fetch(error)?;
            let output: Tensor<T> = run_args.fetch(output)?;
            //TODO: instead of logging these should be accessible within a local class buffer
            //Vec<TrainOutput> or [TrainOutput; configured_recency_buffer_size]

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

    pub fn infer<T: tensorflow::TensorType>(
        &self,
        inputs: &Vec<Vec<T>>,
    ) -> Result<Vec<Tensor<T>>, Box<dyn Error>> {
        let mut result = vec![];
        let mut input_tensor: Tensor<T> = Tensor::new(&[1u64, inputs[0].len() as u64]);
        let mut input_iter = inputs.into_iter();
        let mut i = 0;
        loop {
            i += 1;
            let input = input_iter.next();
            if input.is_none() {
                break;
            }
            let input = input.unwrap();
            // now get input and label as slices
            let input = input.as_slice();
            // now assign the input and label to the tensor
            for i in 0..input.len() {
                input_tensor[i] = input[i].clone();
            }
            let mut run_args = SessionRunArgs::new();
            let output = run_args.request_fetch(&self.Output_op, 0);
            run_args.add_feed(&self.Input, 0, &input_tensor);
            self.session.run(&mut run_args)?;
            let output: Tensor<T> = run_args.fetch(output)?;
            log::info!("input: {:?} output: {:?}", input, output);
            result.push(output);
        }
        Ok(result)
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
    #[test]
    fn test_initial() {
        let mut scope = Scope::new_root_scope();
        let network = vec![
            //layers::std_layer::new(2, activations::Tanh(10)),
            layers::std_layer::new(100, activations::Tanh(10), DataType::Float),
            layers::std_layer::new(100, activations::Tanh(10), DataType::Float),
            layers::std_layer::new(1, activations::Sigmoid(100), DataType::Float),
        ];

        let mut Net = Brain::new(
            "test-initial".to_string(),
            network,
            //TODO: this is recklessly verbose
            Some(Box::new(GradientDescentOptimizer::new(
                ops::constant(0.1 as f32, &mut scope).unwrap(),
            ))),
            2,
            DataType::Float,
            &mut scope,
        )
        .unwrap();

        //train the network
        let mut rrng = rand::thread_rng();
        // create 100 entries for inputs and outputs of xor
        for _ in 0..100 {
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            for _ in 0..10 {
                // instead of the above, generate either 0 or 1 and cast to f32
                let input = vec![(rrng.gen::<u8>() & 1) as f32, (rrng.gen::<u8>() & 1) as f32];
                let output = vec![(input[0] as u8 ^ input[1] as u8) as f32];
                inputs.push(input);
                outputs.push(output);
            }
            let res = Net.train(&inputs, &outputs).unwrap();
            //println!("inputs: {:?} \n res: {:?}", inputs, res);
        }
        //save the model
        let uuid = Uuid::new_v4();
        let save_file = format!("test-initial-{}", uuid);
        Net.save(&save_file).unwrap();
        return;
    }
    #[test]
    fn test_builder() {
        let mut scope = Scope::new_root_scope();
        let mut Net = Brain()
            .name("test-builder")
            .num_inputs(2)
            .input_type(DataType::Float)
            //TODO: input_type(DataType::Float)
            .add_layer(layers::std_layer::new(
                20,
                activations::Sigmoid(1),
                DataType::Float,
            ))
            .add_layer(layers::std_layer::new(
                20,
                activations::Sigmoid(1),
                DataType::Float,
            ))
            .add_layer(layers::std_layer::new(
                1,
                activations::Sigmoid(1),
                DataType::Float,
            ))
            //TODO: there should be a helper function for this, we may create
            //our own wrapper trait here if its better than a trait in tensorflow-rs
            //.optimizer(Box::new(GradientDescentOptimizer::new(
            //    ops::constant(0.001 as f32, &mut scope).unwrap(),
            //)))
            .optimizer(Box::new({
                let mut res = AdadeltaOptimizer::new();
                res.set_learning_rate(ops::constant(1.0 as f32, &mut scope).unwrap());
                res
            }))
            .build(&mut scope)
            .unwrap();

        //train the network
        let mut rrng = rand::thread_rng();
        // create 100 entries for inputs and outputs of xor
        for _ in 0..200 {
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            for _ in 0..200 {
                // instead of the above, generate either 0 or 1 and cast to f32
                let input = vec![(rrng.gen::<u8>() & 1) as f32, (rrng.gen::<u8>() & 1) as f32];
                let output = vec![(input[0] as u8 ^ input[1] as u8) as f32];
                inputs.push(input);
                outputs.push(output);
            }
            assert_eq!(inputs.len(), outputs.len());
            let res = Net.train(&inputs, &outputs).unwrap();
        }
        //print the output of the network
        let mut input: Vec<Vec<f32>> = vec![vec![0.0, 1.0]];
        input.push(vec![1.0, 0.0]);
        input.push(vec![0.0, 0.0]);
        input.push(vec![1.0, 1.0]);

        let output = Net.infer(&input).unwrap();
        println!(
            "XOR test input: {:?} \n XOR test output: {:?}",
            input, output
        );

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
