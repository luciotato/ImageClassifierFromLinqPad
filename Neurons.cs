
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace IC
{
        class Neuron
        {
            public readonly NeuralNet Net;
            public readonly int Layer, Index;

            public double[] InputWeights;
            public double Bias;

            public Activator Activator => Net.Activators[Layer];

            public bool IsOutputNeuron => Layer == Net.Neurons.Length - 1;

            static readonly Random _random = new Random();

            static double GetSmallRandomNumber() =>
                (.0009 * _random.NextDouble() + .0001) * (_random.Next(2) == 0 ? -1 : 1);

        //constructor
        public Neuron(NeuralNet net, int layer, int index, int inputWeightCount)
            {
                Net = net;
                Layer = layer;
                Index = index;

                Bias = GetSmallRandomNumber();
                InputWeights = Enumerable.Range(0, inputWeightCount).Select(_ => GetSmallRandomNumber()).ToArray();
            }
        }

        class NeuralNet
        {
            public readonly Neuron[][] Neurons;     // Layers of neurons
            public Activator[] Activators;          // Activators for each layer

        //constructor
        public NeuralNet(params int[] neuronCountInEachLayer)   // including the input layer
            {
                Neurons = neuronCountInEachLayer
                    .Skip(1)                          // Skip the input layer
                    .Select((count, layerInx) =>
                       Enumerable.Range(0, count)
                                 .Select(index => new Neuron(this, layerInx, index, neuronCountInEachLayer[layerInx]))
                                 .ToArray())
                    .ToArray();
            /*
                for (int layerInx = 1; layerInx < neuronCountInEachLayer.Length; layerInx++) {  // start at 1, Skip the input layer
                    int neuronsInThisLayer = neuronCountInEachLayer[layerInx];
                    Neurons[layerInx] = new Neuron[neuronsInThisLayer]; //create layer
                    for (int neuronIndex = 0; neuronIndex < neuronsInThisLayer; neuronIndex++) {
                        Neurons[layerInx][neuronIndex]= new Neuron(this, layerInx, neuronIndex, neuronCountInEachLayer[layerInx-1]);
                    }
                }
            */

            // Default to ReLU activators
            Activators = Enumerable
                    .Repeat((Activator)new ReLUActivator(), neuronCountInEachLayer.Length - 1)
                    .ToArray();
            }
        }

        class FiringNeuron
        {
            public readonly Neuron Neuron;

            public double TotalInput, Output;
            public double InputVotes, OutputVotes;   // Votes for change = slope of the loss vs input/output

            //constructor
            public FiringNeuron(Neuron neuron) => Neuron = neuron;

            public void ComputeTotalInput(double[] inputValues)
            {
                double sum = 0;

                for (int i = 0; i < Neuron.InputWeights.Length; i++)
                    sum += inputValues[i] * Neuron.InputWeights[i];

                TotalInput = Neuron.Bias + sum;
            }

            public void AdjustWeightsAndBias(double[] inputValues, double learningRate) {            
            //LMT removed: unsafe, reason: not much difference in performance

                double adjustment = InputVotes * learningRate;

                lock (Neuron)
                Neuron.Bias += adjustment;

                int max = Neuron.InputWeights.Length;

                //fixed (double* inputs = inputValues)
                //fixed (double* weights = Neuron.InputWeights)
                    lock (Neuron.InputWeights)
                        for (int i = 0; i < max; i++)
                             Neuron.InputWeights [i] += adjustment * inputValues [i];
                            // Using pointers avoids bounds-checking and so reduces the time spent holding the lock.
                            //*(weights + i) += adjustment * *(inputs + i);
            }
        }

        class FiringNet
        {
            public readonly NeuralNet Net;
            public FiringNeuron[][] Neurons;
            //FiringNeuron[][] NeuronsWithLayersReversed;

            public FiringNeuron[] OutputLayer => Neurons[Neurons.Length - 1];

            public IEnumerable<double> OutputValues => OutputLayer.Select(n => n.Output);

        //constructor
        public FiringNet(NeuralNet net)
            {
                Net = net;
                Neurons = Net.Neurons.Select(layer => layer.Select(n => new FiringNeuron(n)).ToArray()).ToArray();
                //NeuronsWithLayersReversed = Neurons.Reverse().ToArray();
            }

            public void FeedForward(double[] inputValues)
            {
                int i = 0;
                foreach (var layer in Neurons)
                {
                    foreach (var neuron in layer)
                        neuron.ComputeTotalInput(inputValues);

                    Net.Activators[i++].ComputeOutputs(layer);

                    // The outputs for this layer become the inputs for the next layer.
                    if (layer != OutputLayer)
                        inputValues = layer.Select(l => l.Output).ToArray();
                }
            }

            public void Learn(double[] inputValues, double[] desiredOutputs, double learningRate)
            {
                FeedForward(inputValues);

                FiringNeuron[] layerJustProcessed = null;

                // Calculate all the output and input votes.
                //foreach (var layer in NeuronsWithLayersReversed) {
                for (int layerInx=this.Neurons.Count()-1; layerInx>=0; layerInx--) { //start at the output layer and move backwards
                    var layer = Neurons[layerInx];
                    bool isOutputLayer = layerJustProcessed == null;
                    foreach (var neuron in layer)
                    {
                        if (isOutputLayer)
                            // For neurons in the output layer, the loss vs output slope = -error.
                            neuron.OutputVotes = desiredOutputs[neuron.Neuron.Index] - neuron.Output;
                        else
                            // For hidden neurons, the loss vs output slope = weighted sum of next layer's input slopes.
                            neuron.OutputVotes =
                                layerJustProcessed.Sum(n => n.InputVotes * n.Neuron.InputWeights[neuron.Neuron.Index]);

                        // The loss vs input slope = loss vs output slope times activation function slope (chain rule).
                        neuron.InputVotes = neuron.OutputVotes * neuron.Neuron.Activator.GetActivationSlopeAt(neuron);
                    }
                    layerJustProcessed = layer;
                }

                // We can improve the training by scaling the learning rate by the layer index.
                int learningRateMultiplier = Neurons.Length;

                // Updates weights and biases.
                foreach (var layer in Neurons)
                {
                    foreach (var neuron in layer)
                        neuron.AdjustWeightsAndBias(inputValues, learningRate * learningRateMultiplier);

                    if (layer != OutputLayer)
                        inputValues = layer.Select(l => l.Output).ToArray();

                    learningRateMultiplier--;
                }
            }
        }

        class Trainer
        {
            Random _random = new Random();

            public readonly NeuralNet Net;
            public int CurrentEpoch;
            public double CurrentAccuracy;
            //public int Iterations;
            public string TrainingInfo;

            public Trainer(NeuralNet net) => Net = net;

            public void Train(Sample[] trainingData, Sample[] testingData, double learningRate, int epochs)
            {
                _random = new Random();
                Sample[] trainingSet = trainingData; //= trainingData.ToArray();

                TrainingInfo = $"Learning rate = {learningRate}";

                for (CurrentEpoch = 0; CurrentEpoch < epochs; CurrentEpoch++)
                {
                    Stopwatch stw = Stopwatch.StartNew();
                    Console.Write($"Training epoch {CurrentEpoch}... ");
                    CurrentAccuracy = TrainEpoch(trainingSet, learningRate);
                    stw.Stop();
                    Console.WriteLine();
                    learningRate *= .9;   // This help to avoids oscillation as our accuracy improves.
                    Console.WriteLine($"Done {stw.ElapsedMilliseconds}ms - Training accuracy = {CurrentAccuracy.ToString("N1")}%");
            }

                string testAccuracy = ((Test(new FiringNet(Net), testingData) * 100).ToString("N1") + "%").Dump("% success with testing data");
                TrainingInfo += $"\r\nTotal epochs = {CurrentEpoch}\r\nFinal test accuracy = {testAccuracy}";
            }

        public double TrainEpoch(Sample[] trainingData, double learningRate) {
            Shuffle(_random, trainingData);   // For each training epoch, randomize order of the training samples.

            // One FiringNet per thread to avoid thread-safety problems.
            var trainer = new ThreadLocal<FiringNet>(() => new FiringNet(Net));

            Parallel.ForEach(trainingData, CancellableParallel, sample => {
                trainer.Value.Learn(sample.Data, sample.ExpectedOutput, learningRate);
                //Interlocked.Increment(ref Iterations);
            });

            /*foreach(var sample in trainingData) {
                trainer.Value.Learn(sample.Data, sample.ExpectedOutput, learningRate);
                //Interlocked.Increment(ref Iterations);
            }
            */

                return Test(new FiringNet(Net), trainingData.Take(10000).ToArray()) * 100;
            }

            public double Test(FiringNet firingNet, Sample[] samples)
            {
                int bad = 0, good = 0;
                foreach (var sample in samples)
                {
                    firingNet.FeedForward(sample.Data);
                    if (sample.IsOutputCorrect(firingNet.OutputValues.ToArray()))
                        good++;
                    else
                        bad++;
                }
                return (double)good / (good + bad);
            }

            static void Shuffle<T>(Random random, T[] array)
            {
                int n = array.Length;
                while (n > 1)
                {
                    int k = random.Next(n--);
                    T temp = array[n];
                    array[n] = array[k];
                    array[k] = temp;
                }
            }

            // We want to cancel any outstanding training when the user cancels or re-runs the query.
            CancellationTokenSource _cancelSource = new CancellationTokenSource();
            ParallelOptions CancellableParallel => new ParallelOptions { CancellationToken = _cancelSource.Token };
            //Trainer() => Util.Cleanup += (sender, args) => _cancelSource.Cancel();

            //object ToDump() => NeuralNetRenderer(this);
        }

}

