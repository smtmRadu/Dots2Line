using System;
using UnityEditor;
using UnityEngine;
using static NeuroForge.Functions;

namespace NeuroForge
{

    [Serializable]
    public class NeuralNetwork : ScriptableObject
    {
        [SerializeField] public int[] layerFormat;
        [SerializeField] public NeuronLayer[] neuronLayers;
        [SerializeField] public WeightLayer[] weightLayers;
        [SerializeField] public BiasLayer[] biasLayers;

        [SerializeField] public InitializationType initializationType; // shows how the network was initialized
        [SerializeField] public ActivationType activationType;
        [SerializeField] public ActivationType outputActivationType;
        [SerializeField] public LossType lossType;

        private WeightLayer[] weightGradients;
        private WeightLayer[] weightMomentums;
        private BiasLayer[] biasGradients;
        private BiasLayer[] biasMomentums;

        int backpropsCount = 0;
        static object lockBC = new object();

        /// <summary>
        /// MLP neural network. Uses SGD with momentum. 
        /// </summary>
        /// <param name="inp">number of inputs</param>
        /// <param name="outp">number of outputs</param>
        /// <param name="hidUnits">number of hidden neurons per hidden layer</param>
        /// <param name="layNum">number of hidden layers</param>
        /// <param name="activFunc">activation type used for hidden neurons</param>
        /// <param name="outActivFunc">activation type used for output neurons</param>
        /// <param name="loss">loss function used for backpropagation</param>
        /// <param name="initType">weight initialization</param>
        /// <param name="createAsset">create the ANN asset?</param>
        /// <param name="name">name of the ANN asset</param>
        public NeuralNetwork(int inp, int outp, int hidUnits, int layNum,
                             ActivationType activFunc, ActivationType outActivFunc, LossType loss,
                             InitializationType initType, bool createAsset, string name = "ann")
        {
            this.layerFormat = GetFormat(inp, outp, hidUnits, layNum);
            this.initializationType = initType;
            this.activationType = activFunc;
            this.outputActivationType = outActivFunc;
            this.lossType = loss;

            neuronLayers = new NeuronLayer[layerFormat.Length];
            biasLayers = new BiasLayer[layerFormat.Length];
            weightLayers = new WeightLayer[layerFormat.Length - 1];

            for (int i = 0; i < neuronLayers.Length; i++)
            {
                if (i != neuronLayers.Length - 1)
                    neuronLayers[i] = new NeuronLayer(layerFormat[i], activationType);
                else
                    neuronLayers[i] = new NeuronLayer(layerFormat[i], outputActivationType);
                biasLayers[i] = new BiasLayer(layerFormat[i], initType);

            }
            for (int i = 0; i < neuronLayers.Length - 1; i++)
            {
                weightLayers[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], initType);
            }

            if (createAsset)
            {
                Debug.Log(name + " was created!");
                AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
                AssetDatabase.SaveAssets();
            }
        }


        public double[] Forward(double[] inputs, NeuronLayer[] thread_neurons = null)
        {
            if (thread_neurons == null)
                thread_neurons = neuronLayers;

            thread_neurons[0].SetOutValues(inputs);
            for (int l = 1; l < thread_neurons.Length; l++)
            {
                for (int n = 0; n < thread_neurons[l].neurons.Length; n++)
                {
                    double sumValue = biasLayers[l].biases[n];
                    for (int prevn = 0; prevn < thread_neurons[l - 1].neurons.Length; prevn++)
                    {
                        sumValue += thread_neurons[l - 1].neurons[prevn].OutValue * weightLayers[l - 1].weights[prevn][n];
                    }
                    thread_neurons[l].neurons[n].InValue = sumValue;
                }

                thread_neurons[l].Activate();
            }
            return thread_neurons[thread_neurons.Length - 1].GetOutValues();
        }
        public double Backward(double[] inputs, double[] labels, bool parallel = false)
        {
            // Initialize thread neurons
            NeuronLayer[] THREAD_nl = neuronLayers;
            if (parallel)
            {
                THREAD_nl = new NeuronLayer[neuronLayers.Length];
                for (int i = 0; i < THREAD_nl.Length; i++)
                {
                    THREAD_nl[i] = neuronLayers[i].Clone() as NeuronLayer;
                }
            }

            // Initialize gradients
            if (weightGradients == null || weightGradients.Length < 1)
                ZeroGrad();

            Forward(inputs, THREAD_nl);

            double error = CalculateLayerCost_output(labels, THREAD_nl);

            for (int wLayer = weightLayers.Length - 1; wLayer >= 0; wLayer--)
            {
                UpdateGradients(weightGradients[wLayer], biasGradients[wLayer + 1], THREAD_nl[wLayer], THREAD_nl[wLayer + 1]);
                CalculateLayerCost(THREAD_nl[wLayer], weightLayers[wLayer], THREAD_nl[wLayer + 1]);
            }

            // backpropsCount++ is applied inside UpdateGradients method
            return error;
        }

        public void GradClipNorm(float threshold)
        {
            double global_sum = 0;

            // Sum weights' gradients
            foreach (var grad_layer in weightGradients)
            {
                foreach (var clump in grad_layer.weights)
                {
                    foreach (var w_grad in clump)
                    {
                        global_sum += w_grad * w_grad;
                    }
                }
            }

            // Sum biases' gradients
            foreach (var bias_layer in biasGradients)
            {
                foreach (var b_grad in bias_layer.biases)
                {
                    global_sum += b_grad * b_grad;
                }
            }

            double scalar = threshold / Math.Max(threshold, global_sum);

            // Normalize weights
            for (int lay = 0; lay < weightGradients.Length; lay++)
            {
                for (int i = 0; i < weightGradients[lay].weights.Length; i++)
                {
                    for (int j = 0; j < weightGradients[lay].weights[i].Length; j++)
                    {
                        weightGradients[lay].weights[i][j] *= scalar;
                    }
                }
            }

            // Normalize biases
            for (int lay = 0; lay < biasGradients.Length; lay++)
            {
                for (int i = 0; i < biasGradients[lay].biases.Length; i++)
                {
                    biasGradients[lay].biases[i] *= scalar;
                }
            }
        }
        public void OptimStep(float learn_rate, float momentum, float regularization)
        {
            learn_rate /= backpropsCount;
            backpropsCount = 0;

            double weightDecay = 1 - regularization * learn_rate;
            for (int l = 0; l < weightLayers.Length; l++)
            {
                for (int i = 0; i < weightLayers[l].weights.Length; i++)
                {
                    for (int j = 0; j < weightLayers[l].weights[i].Length; j++)
                    {
                        double weight = weightLayers[l].weights[i][j];
                        double veloc = weightMomentums[l].weights[i][j] * momentum - weightGradients[l].weights[i][j] * learn_rate;

                        weightMomentums[l].weights[i][j] = veloc;
                        weightLayers[l].weights[i][j] = weight * weightDecay + veloc;

                        //Reset the gradient
                        weightGradients[l].weights[i][j] = 0;
                    }
                }
            }
            for (int i = 0; i < biasLayers.Length; i++)
            {
                for (int j = 0; j < biasLayers[i].biases.Length; j++)
                {
                    double veloc = biasMomentums[i].biases[j] * momentum - biasGradients[i].biases[j] * learn_rate;

                    biasMomentums[i].biases[j] = veloc;
                    biasLayers[i].biases[j] += veloc;

                    biasGradients[i].biases[j] = 0;
                }
            }
        }
        public void Save()
        {
            EditorUtility.SetDirty(this);
            AssetDatabase.SaveAssetIfDirty(this);
        }
        public void ZeroGrad()
        {
            biasGradients = new BiasLayer[layerFormat.Length];
            biasMomentums = new BiasLayer[layerFormat.Length];
            weightGradients = new WeightLayer[layerFormat.Length - 1];
            weightMomentums = new WeightLayer[layerFormat.Length - 1];

            for (int i = 0; i < neuronLayers.Length; i++)
            {
                biasGradients[i] = new BiasLayer(layerFormat[i], InitializationType.Zero);
                biasMomentums[i] = new BiasLayer(layerFormat[i], InitializationType.Zero);

            }
            for (int i = 0; i < weightLayers.Length; i++)
            {
                weightGradients[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], InitializationType.Zero);
                weightMomentums[i] = new WeightLayer(neuronLayers[i], neuronLayers[i + 1], InitializationType.Zero);
            }
        }

        private void UpdateGradients(WeightLayer weightGradient, BiasLayer biasGradient, NeuronLayer previousNeuronLayer, NeuronLayer nextNeuronLayer)
        {
            lock (weightGradient)
            {
                backpropsCount++;
                for (int i = 0; i < previousNeuronLayer.neurons.Length; i++)
                {
                    for (int j = 0; j < nextNeuronLayer.neurons.Length; j++)
                    {
                        weightGradient.weights[i][j] += previousNeuronLayer.neurons[i].OutValue * nextNeuronLayer.neurons[j].CostValue;
                    }
                }
            }
            lock (biasGradient)
            {
                for (int i = 0; i < nextNeuronLayer.neurons.Length; i++)
                {
                    biasGradient.biases[i] += 1 * nextNeuronLayer.neurons[i].CostValue;
                }
            }
        }
        private void CalculateLayerCost(NeuronLayer layer, WeightLayer weights, NeuronLayer nextLayer)
        {
            for (int i = 0; i < layer.neurons.Length; i++)
            {
                double costVal = 0;
                for (int j = 0; j < nextLayer.neurons.Length; j++)
                {
                    costVal += nextLayer.neurons[j].CostValue * weights.weights[i][j];
                }
                costVal *= Derivative.DeriveValue(layer.neurons[i].InValue, activationType);

                layer.neurons[i].CostValue = costVal;
            }
        }
        private double CalculateLayerCost_output(double[] labels, NeuronLayer[] neurLays)
        {
            NeuronLayer outLayer = neurLays[neuronLayers.Length - 1];
            double err = 0;
            if (outputActivationType != ActivationType.SoftMax)
            {
                for (int i = 0; i < outLayer.neurons.Length; i++)
                {
                    switch (lossType)
                    {
                        case LossType.MeanSquare:
                            outLayer.neurons[i].CostValue = Loss.MeanSquareDerivative(outLayer.neurons[i].OutValue, labels[i]) * Derivative.DeriveValue(outLayer.neurons[i].InValue, outputActivationType);
                            err += Error.MeanSquare(outLayer.neurons[i].OutValue, labels[i]);
                            break;
                        case LossType.CrossEntropy:
                            outLayer.neurons[i].CostValue = Loss.CrossEntropyDerivative(outLayer.neurons[i].OutValue, labels[i]) * Derivative.DeriveValue(outLayer.neurons[i].InValue, outputActivationType);
                            err += Error.CrossEntropy(outLayer.neurons[i].OutValue, labels[i]);
                            break;
                        case LossType.MeanAbsolute:
                            outLayer.neurons[i].CostValue = Loss.AbsoluteDerivative(outLayer.neurons[i].OutValue, labels[i]) * Derivative.DeriveValue(outLayer.neurons[i].InValue, outputActivationType);
                            err += Error.MeanAbsolute(outLayer.neurons[i].OutValue, labels[i]);
                            break;
                    }
                }
            }
            else
            {
                double[] derivedInValuesBySoftMax = new double[labels.Length];
                for (int i = 0; i < derivedInValuesBySoftMax.Length; i++)
                    derivedInValuesBySoftMax[i] = outLayer.neurons[i].InValue;

                Derivative.SoftMax(derivedInValuesBySoftMax);

                for (int i = 0; i < outLayer.neurons.Length; i++)
                {
                    switch (lossType)
                    {
                        case LossType.MeanSquare:
                            outLayer.neurons[i].CostValue = Loss.MeanSquareDerivative(outLayer.neurons[i].OutValue, labels[i]) * derivedInValuesBySoftMax[i];
                            err += Error.MeanSquare(outLayer.neurons[i].OutValue, labels[i]);
                            break;
                        case LossType.CrossEntropy:
                            outLayer.neurons[i].CostValue = Loss.CrossEntropyDerivative(outLayer.neurons[i].OutValue, labels[i]) * derivedInValuesBySoftMax[i];
                            err += Error.CrossEntropy(outLayer.neurons[i].OutValue, labels[i]);
                            break;
                        case LossType.MeanAbsolute:
                            outLayer.neurons[i].CostValue = Loss.AbsoluteDerivative(outLayer.neurons[i].OutValue, labels[i]) * derivedInValuesBySoftMax[i];
                            err += Error.MeanAbsolute(outLayer.neurons[i].OutValue, labels[i]);
                            break;
                    }
                }



            }

            return err / labels.Length;
        }

        // OTHER
        static int[] GetFormat(int inputs, int outs, int hidden_units, int hidden_lay_num)
        {
            int[] form = new int[2 + hidden_lay_num];

            form[0] = inputs;
            for (int i = 1; i <= hidden_lay_num; i++)
            {
                form[i] = hidden_units;
            }
            form[form.Length - 1] = outs;

            return form;
        }
        public int GetNoInputs() => layerFormat[0];
        public int GetNoOutputs() => layerFormat[layerFormat.Length - 1];
    }


}