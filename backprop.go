package main

import "math"

const startingCoefficient float64 = 3.0
const coeffDecay float64 = 0.98

type BackPropogator struct {
  nn NeuronNetwork
  expectedOutput float64
  iteration int
}

func (bp *BackPropogator) Propogate(output float64) {
  err := (bp.expectedOutput - output)
  coeff := startingCoefficient * math.Pow(coeffDecay, float64(bp.iteration))

  errors := [3][3]float64{}

  // Error Calculation: d3 = w34d4 + w35d5
  for i := len(bp.nn.neuronLayers) - 1; i >= 0; i-- {
    errors[i] = [3]float64{}
    for j := 0; j < len(bp.nn.neuronLayers[i].neurons); j++ {
      if i == len(bp.nn.neuronLayers) -1 { // Top layer, use err instead
        errors[i][j] = err
      } else {
        for k := 0; k < len(bp.nn.neuronLayers[i+1].neurons); k++ {
          if k == 0 {
            errors[i][j] = 0
          }
          errors[i][j] += bp.nn.neuronLayers[i+1].neurons[k].weights[j] * errors[i+1][k]
        }
      }
    }
  }

  // Weight Adjustment: w'12 = w12 + ( h * d * df1(e)/de * y2)
  for i := 0; i < len(bp.nn.neuronLayers); i++ {
    for j := 0; j < len(bp.nn.neuronLayers[i].neurons); j ++ {
      for k := 0; k < len(bp.nn.neuronLayers[i].neurons[j].weights); k++ {
        input := 0.0
        if i == 0 {
          input = bp.nn.inputs[k]
        } else {
          input = bp.nn.neuronLayers[i - 1].neurons[k].output
        }
        bp.nn.neuronLayers[i].neurons[j].weights[k] += coeff * errors[i][j] * math.Atan(input) * bp.nn.neuronLayers[i].neurons[j].output
      }
    }
  }

  bp.iteration++
}
