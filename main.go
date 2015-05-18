package main

import "fmt"
import "math"
import "os"
import "strconv"

func main() {
  var iterations int
  var e error
  iterations, e = strconv.Atoi(os.Args[len(os.Args) -1])
  if e != nil {
    fmt.Println("Parse error")
  }
  inputs := []float64{1.1, 0.8}
  target := 0.6
  tests := 1
  sum := 0.0

  for i := 0; i < tests; i++ {
    res := backPropSimulation(SetupNeuralNetwork(), iterations, inputs, target)
      sum += math.Abs(res - target) / target
      fmt.Println("Result:",res)
    }
    avgErr := sum / float64(tests) // In the future, std deviation may be more useful

    fmt.Println(iterations, "iterations completed with", (avgErr * 100), "% error")
}

func backPropSimulation(nn NeuronNetwork, iterations int, inputs []float64, target float64) float64 {
  prop := BackPropogator{nn, target, 1}

  for i := 0; i < iterations; i++ {
    prop.nn.inputs = inputs

    prop.nn.Update() // Execute network
    output := prop.nn.outputs[0]
    prop.Propogate(output) // Run backpropagation algorithm given our output
  }

  return prop.nn.outputs[0]
}
