package main

import "math"

const neuralLayers = 2;
const outputNodes int = 1;
const middleNodes int = 3;
const inputNodes int = 2;

const initialWeight float64 = 0.3;
const initialThreshold float64 = 1.0;

func SetupNeuralNetwork() NeuronNetwork {
  layerSizes := []int{inputNodes,middleNodes,outputNodes}

  layers := make([]NeuronLayer, neuralLayers)
  for i := 1; i < len(layerSizes); i++ { // Start at 1, our inputs are not neural nodes
    neurons := make([]Neuron, layerSizes[i])
    for k := 0; k < layerSizes[i]; k++ {
      weights := make([]float64, layerSizes[i-1])
      for j := 0; j < layerSizes[i-1]; j++ {
        weights[j] = initialWeight
      }
      neurons[k] = Neuron{weights, initialThreshold, 0}
    }
    layers[i-1] = NeuronLayer{neurons}
  }
  network := NeuronNetwork{[]float64{0,0}, []float64{0}, layers}

  return network
}

// Neuron
type Neuron struct {
  weights []float64 // Weights to be applied to inputs
  threshold, output float64 // Threshold for linear function and output value
}

func (n *Neuron) WeightedSigma(inputs []float64) float64 { // Sums inputs * weights
  var sum float64 = 0
  for i := 0; i < len(inputs); i++ {
    sum += n.weights[i] * inputs[i]
  }
  return sum;
}

func (n *Neuron) NonLinearFunc(value float64) float64 { // Applies non-linear function with threshold
  return math.Tan(value) + n.threshold
}

func (n *Neuron) Update(inputs []float64) { // Computes output of node for given inputs
  sum := n.WeightedSigma(inputs)
  n.output = n.NonLinearFunc(sum)
}


// Neuron Layer
type NeuronLayer struct {
  neurons []Neuron
}

func (nl *NeuronLayer) Update(inputs []float64) { // Computes output of each node in layer
  for i := 0; i < len(nl.neurons); i++ {
    nl.neurons[i].Update(inputs)
  }
}

// Neuron Network
type NeuronNetwork struct {
  inputs, outputs []float64 // Inputs and outputs of network
  neuronLayers []NeuronLayer
}

func (nn *NeuronNetwork) Update() { // Computes output of each layer for given input and updates the output of the network
  // Run network updates
  inputs := nn.inputs
  for i := 0; i < len(nn.neuronLayers); i++ {
    nn.neuronLayers[i].Update(inputs)
    layerSize := len(nn.neuronLayers[i].neurons)
    inputs = make([]float64, layerSize, layerSize)
    for j := 0; j < layerSize; j++ {
      inputs[j] = nn.neuronLayers[i].neurons[j].output
    }
  }

  nn.outputs = inputs
}
