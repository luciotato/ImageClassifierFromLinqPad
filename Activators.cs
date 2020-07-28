using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IC
{

    abstract class Activator
    {
        public abstract void ComputeOutputs(FiringNeuron[] layer);
        public abstract double GetActivationSlopeAt(FiringNeuron neuron);
    }

    class ReLUActivator : Activator
    {
        public override void ComputeOutputs(FiringNeuron[] layer)
        {
            foreach (var neuron in layer)
                neuron.Output = neuron.TotalInput > 0 ? neuron.TotalInput : neuron.TotalInput / 100;
        }

        public override double GetActivationSlopeAt(FiringNeuron neuron) => neuron.TotalInput > 0 ? 1 : .01;
    }

    class LogisticSigmoidActivator : Activator
    {
        public override void ComputeOutputs(FiringNeuron[] layer)
        {
            foreach (var neuron in layer)
                neuron.Output = 1 / (1 + Math.Exp(-neuron.TotalInput));
        }

        public override double GetActivationSlopeAt(FiringNeuron neuron)
            => neuron.Output * (1 - neuron.Output);
    }

    class HyperTanActivator : Activator
    {
        public override void ComputeOutputs(FiringNeuron[] layer)
        {
            foreach (var neuron in layer)
                neuron.Output = Math.Tanh(neuron.TotalInput);
        }

        public override double GetActivationSlopeAt(FiringNeuron neuron)
        {
            var tanh = neuron.Output;
            return 1 - tanh * tanh;
        }
    }

    class SoftMaxActivator : Activator
    {
        public override void ComputeOutputs(FiringNeuron[] layer)
        {
            double sum = 0;

            foreach (var neuron in layer)
            {
                neuron.Output = Math.Exp(neuron.TotalInput);
                sum += neuron.Output;
            }

            foreach (var neuron in layer)
            {
                var oldOutput = neuron.Output;
                neuron.Output = neuron.Output / (sum == 0 ? 1 : sum);
            }
        }

        public override double GetActivationSlopeAt(FiringNeuron neuron)
        {
            double y = neuron.Output;
            return y * (1 - y);
        }
    }

    class SoftMaxActivatorWithCrossEntropyLoss : SoftMaxActivator  // Use this only on the output layer!
    {
        // This is the derivative after modifying the loss function.
        public override double GetActivationSlopeAt(FiringNeuron neuron) => 1;
    }

}
