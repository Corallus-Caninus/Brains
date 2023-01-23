# Brains
NOTE: this project is very early and was spawned from my own architecture research. contributions, opinions and issues are very welcome! Version 1.X.X will be considered ready for public use.

an Artificial Neural Network framework built on Tensorflow-rs bindings for creating architectures similar to keras but also with direct integration for custom layers in low level Tensorflow. Includes native checkpointing, inference, batch trainning and iterative trainning. See the unittests to get an idea of how things are called until documentation is created. Also ensure to enable or disable the GPU feature for tensorflow if you want to offload computation.

Currently all inputs and outputs are represented as flattend 1D rust Vecs.

once the following TODO is finished this will be updated on crates and can be considered a fledgling framework (v1.X.X):
# TODO: 
Refactor everything to use traits and the builder pattern to get a more Keras sequential API user interface and something that can be built upon and implemented more easily onto other frameworks later (e.g. libtorch).

create a dev Branch and pull into Master as a Release branch

Documentation. Docstrings and a project overview landing page that discusses the Mods.


Please let me know what you think: ward.joshua92@yahoo.com
