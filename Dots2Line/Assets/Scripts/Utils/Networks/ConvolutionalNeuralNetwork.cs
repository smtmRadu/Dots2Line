using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace NeuroForge
{
    [Serializable]
    public class ConvolutionalNeuralNetwork : ScriptableObject, ISerializationCallbackReceiver
    {
        [SerializeField] public NeuralNetwork network;
        [SerializeField] private int convolutionLevel;

        private float[,] kernel;
        [SerializeField] private KernelType kernelType = KernelType.Laplac_3x3;
        [SerializeField] private PaddingType paddingType = PaddingType.Mirror;
        [SerializeField] private PoolType poolType = PoolType.Max;

        /// <summary>
        /// CNN with heightmap convolution.
        /// Laplacian 3x3 kernel, Max pooling, ReLU activ, CE loss, HE init.
        /// </summary>
        /// <param name="i_w">image width</param>
        /// <param name="i_h">image height</param>
        /// <param name="outs">number of obj classes/outputs</param>
        /// <param name="convLvl">convolution strength</param>
        /// <param name="hidUnits">number of hidden neurons per hidden layer</param>
        /// <param name="layNum">number of hidden layers</param>
        /// <param name="createAsset">create the CNN asset + ANN asset?</param>
        /// <param name="name">name of the CNN asset</param>
        public ConvolutionalNeuralNetwork(int i_w, int i_h, int outs, int convLvl, int hidUnits, int layNum, bool createAsset = true, string name = "cnn")
        {
            convolutionLevel = convLvl;

            for (int i = 0; i < convLvl; i++)
            {
                i_w /= 2;
                i_h /= 2;
            }

            network = new NeuralNetwork(i_w * i_h, outs, hidUnits, layNum,
                                        ActivationType.Relu, ActivationType.SoftMax, LossType.CrossEntropy,
                                        InitializationType.He, true, name + "_aux");

            this.kernel = Functions.Image.Kernels.GetKernel(kernelType);

            if (createAsset)
            {
                Debug.Log(name + " was created!");
                // AssetDatabase.CreateAsset(this, "Assets/" + name + ".asset");
                // AssetDatabase.SaveAssets();
            }
        }
        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            this.kernel = Functions.Image.Kernels.GetKernel(kernelType);
        }

        public double[] Forward(float[,] input_image)
        {
            for (int i = 0; i < convolutionLevel; i++)
            {
                Pad(ref input_image);
                Filter(ref input_image);
                RescaleFilteredImage(ref input_image);
                Pool(ref input_image);
            }
            double[] flat_input = FlatMatrix(input_image);
            double[] outputs = network.Forward(flat_input);
            return outputs;
        }
        public double Backward(float[,] input_image, int label, bool parallel = false)
        {
            for (int i = 0; i < convolutionLevel; i++)
            {
                Pad(ref input_image);
                Filter(ref input_image);
                RescaleFilteredImage(ref input_image);
                Pool(ref input_image);
            }

            double[] flat_inputs = FlatMatrix(input_image);
            double[] labels = new double[network.GetNoOutputs()];
            labels[label] = 1;

            double error = network.Backward(flat_inputs, labels, parallel);

            return error;
        }


        public void GradClipNorm(float threshold) => network.GradClipNorm(threshold);
        public void OptimStep(float learnRate, float momentum, float regularization) => network.OptimStep(learnRate, momentum, regularization);
        // public void Save()
        // {
        //     EditorUtility.SetDirty(network);
        //     AssetDatabase.SaveAssetIfDirty(network);
        //     EditorUtility.SetDirty(this);
        //     AssetDatabase.SaveAssetIfDirty(this);
        // }

        // Convolution Methods
        private void Pad(ref float[,] image)
        {
            int padding = 1;
            // ZERO PADDING
            float[,] padded_image = new float[image.GetLength(0) + 2 * padding, image.GetLength(1) + 2 * padding];

            for (int i = 0; i < image.GetLength(0); i++)
            {
                for (int j = 0; j < image.GetLength(1); j++)
                {
                    padded_image[i + 1, j + 1] = image[i, j];
                }
            }

            if (paddingType == PaddingType.Mirror)
            {
                int w = padded_image.GetLength(0);
                int h = padded_image.GetLength(1);

                //Corners
                padded_image[0, 0] = padded_image[1, 1];
                padded_image[0, h - 1] = padded_image[1, h - 2];
                padded_image[w - 1, 0] = padded_image[w - 2, 1];
                padded_image[w - 1, h - 1] = padded_image[w - 2, h - 2];

                //vertical
                for (int i = 1; i < padded_image.GetLength(0) - 1; i++)
                {
                    padded_image[i, 0] = padded_image[i, 1];
                    padded_image[i, h - 1] = padded_image[i, h - 2];
                }
                //horizontal
                for (int j = 1; j < padded_image.GetLength(1) - 1; j++)
                {
                    padded_image[0, j] = padded_image[1, j];
                    padded_image[w - 1, j] = padded_image[w - 2, j];
                }
            }

            image = padded_image;
        }
        private void Filter(ref float[,] image)
        {
            int padding = 1;
            int stripe = 1;

            // Filtering does not affect the dimension of the final image (only pooling)
            // Image is padded. When applying kernel, f_img will be 2 less for each dimension
            float[,] filtered_image = new float[image.GetLength(0) - 2 * padding, image.GetLength(1) - 2 * padding];

            // Parse each pixel
            for (int i = 1; i < image.GetLength(0) - 1; i += stripe)
            {
                for (int j = 1; j < image.GetLength(1) - 1; j += stripe)
                {
                    // Filter-up
                    float sum = 0;
                    for (int k_i = 0; k_i < kernel.GetLength(0); k_i++)
                    {
                        for (int k_j = 0; k_j < kernel.GetLength(1); k_j++)
                        {
                            sum += image[i - 1 + k_i, j - 1 + k_j] * kernel[k_i, k_j];
                        }
                    }

                    filtered_image[i - 1, j - 1] = sum;
                }
            }

            image = filtered_image;
        }
        private void RescaleFilteredImage(ref float[,] image)
        {
            float max_val = float.MinValue;
            float min_val = float.MaxValue;

            // Find min & max
            for (int i = 0; i < image.GetLength(0); i++)
            {
                for (int j = 0; j < image.GetLength(1); j++)
                {
                    if (image[i, j] > max_val)
                        max_val = image[i, j];
                    if (image[i, j] < min_val)
                        min_val = image[i, j];
                }
            }

            // Scale [0,1]
            float delta = max_val - min_val;
            for (int i = 0; i < image.GetLength(0); i++)
            {
                for (int j = 0; j < image.GetLength(1); j++)
                {
                    image[i, j] = (image[i, j] - min_val) / delta;
                }
            }
        }
        private void Pool(ref float[,] image)
        {
            float[,] pooled_image = new float[image.GetLength(0) / 2, image.GetLength(1) / 2];
            for (int i = 0; i < pooled_image.GetLength(0); i++)
            {
                for (int j = 0; j < pooled_image.GetLength(1); j++)
                {
                    float[] local_pool = new float[4];
                    local_pool[0] = image[i * 2, j * 2];
                    local_pool[1] = image[i * 2, j * 2 + 1];
                    local_pool[2] = image[i * 2 + 1, j * 2];
                    local_pool[3] = image[i * 2 + 1, j * 2 + 1];

                    pooled_image[i, j] = poolType == PoolType.Max ? local_pool.Max() : local_pool.Average();
                }
            }

            image = pooled_image;
        }

        private double[] FlatMatrix(float[,] mat)
        {
            int width = mat.GetLength(0);
            int height = mat.GetLength(1);

            double[] flat = new double[width * height];

            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    flat[x * height + y] = (double)mat[x, y];
                }
            }
            return flat;
        }
    }

}
