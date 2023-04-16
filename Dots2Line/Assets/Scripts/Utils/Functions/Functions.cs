using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace NeuroForge
{
    public class Functions
    {
        private static System.Random rand01 = new System.Random();
        public static double randomValue { get { lock (rand01) return rand01.NextDouble(); } }

        public static double RandomGaussian(double mean = 0, double stddev = 1)
        {
            lock (rand01)
            {
                double x1 = 1 - rand01.NextDouble();
                double x2 = 1 - rand01.NextDouble();

                double y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
                return y1 * stddev + mean;
            }
        }
        public static double RandomRange(double minInclusive, double maxExclusive)
        {
            lock(rand01)
                return rand01.NextDouble() * (maxExclusive - minInclusive) + minInclusive;
        }
        public static int RandomRange(int minInclusive, int maxExclusive)
        {
            lock (rand01)
                return rand01.Next(minInclusive, maxExclusive);
        }
        public static T RandomIn<T>(IEnumerable<T> values, List<float> unormProbs = null)
        {
            if (unormProbs == null)
            {
                int randIndex = RandomRange(0, values.Count());
                return values.ElementAtOrDefault(randIndex);
            }
            // recommended to let it as it is
            for (int i = 0; i < unormProbs.Count; i++)
            {
                if (unormProbs[i] <= 0)
                    unormProbs[i] = 1e-8f;
            }

            float random = (float) rand01.NextDouble() * unormProbs.Sum();
            int index = 0; // dont modify this and that -1 at the end, is case for 0 0 0 0 on probs

            while (random > 0)
            {
                random -= unormProbs[index];
                index++;
            }

            return values.ElementAt(index - 1);

        }
        public static void Normalize(IEnumerable<double> values)
        {
            // Calculate mean
            double mean = values.Average();

            // Calculate std
            double sum = 0;
            foreach (var item in values)
            {
                sum += (item - mean) * (item - mean);
            }
            double variance = sum / values.Count();
            double std = Math.Sqrt(variance);

            // Normalize list
            values = values.Select(x => (x - mean) / (std + 1e-8)).ToList();
        }
        public static void Normalize01(IEnumerable<double> list)
        {
            double min = list.Min();
            double max = list.Max();
            list = list.Select(x => (x - min) / (max - min));
        }
        public static void Shuffle<T>(List<T> list)
        {
            lock (rand01)
            {
                for (int i = 0; i < list.Count; i++)
                {
                    int j = rand01.Next(0, list.Count - 1);
                    T temp = list[i];
                    list[i] = list[j];
                    list[j] = temp;
                }
            }
        }
        public static string StringOf(IEnumerable array, string tag = null)
        {
            StringBuilder sb = new StringBuilder();
            if (tag != null) sb.Append(tag);
            sb.Append("[ ");
            foreach (var item in array)
            {
                sb.Append(item.ToString());
                sb.Append(", ");
            }
            sb.Remove(sb.Length - 2, 1);
            sb.Append("]");
            return sb.ToString();
        }
        public static bool IsValueIn<T>(T value, IEnumerable<T> collection)
        {
            foreach (var item in collection)
            {
                if(value.Equals(item)) return true;
            }
            return false;
        }
        public static void DebugInFile(string text, bool newLine = true)
        {
            using (StreamWriter sw = new StreamWriter("C:\\Users\\X\\Desktop\\debug.txt", true))
            {
                if (newLine)
                    sw.WriteLine(text);
                else
                    sw.Write(text);
            }
        }
        public static string HexOf(Color color)
        {
            int r = Mathf.RoundToInt(color.r * 255.0f);
            int g = Mathf.RoundToInt(color.g * 255.0f);
            int b = Mathf.RoundToInt(color.b * 255.0f);

            return string.Format("#{0:X2}{1:X2}{2:X2}", r, g, b);
        }
        public static bool HasNaN(IEnumerable<double> array)
        {
            foreach (var item in array)
            {
                if (double.IsNaN(item)) return true;
            }
            return false;
        }
        public static void Swap<T>(ref T obj1, ref T obj2)
        {
            T temp = obj1;
            obj1 = obj2;
            obj2 = temp;
        }
        public static int ArgMax(double[] values)
        {
            int index = -1;
            double max = double.MinValue;
            for (int i = 0; i < values.Length; i++)
                if (values[i] > max)
                {
                    max = values[i];
                    index = i;
                }
            return index;
        }
        public static T[] FlatOf<T>(T[,] matrix)
        {
            int w = matrix.GetLength(0);
            int h = matrix.GetLength(1);

            T[] flat = new T[w * h];
            int ind = 0;
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    flat[ind++] = matrix[j, i];
                }
            }

            return flat;
        }
        public static T[,] MatrixOf<T>(T[] flat, int w, int h)
        {
            T[,] matrix = new T[w,h];
            int ind = 0;
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    matrix[j, i] = flat[ind++];
                }
            }
            return matrix;
        }

        public readonly struct Activation
        {
            public static double ActivateValue(double value, ActivationType activationFunction)
            {
                switch(activationFunction)
                {
                    case ActivationType.Tanh:
                        return TanH(value);
                    case ActivationType.Sigmoid:
                        return Sigmoid(value);
                    case ActivationType.Relu:
                        return ReLU(value);
                    case ActivationType.LeakyRelu:
                        return LeakyReLU(value);
                    case ActivationType.Silu:
                        return SiLU(value);
                    case ActivationType.ELU:
                        return ELU(value);
                    case ActivationType.SoftPlus:
                        return SoftPlus(value);
                    default: //Linear
                        return value;
                        
                }
                
            }

            public static double Sigmoid(double value) => 1.0 / (1.0 + Math.Exp(-value)); 
            public static double TanH(double value) => Math.Tanh(value);
            public static double ReLU(double value) => Math.Max(0, value);
            public static double LeakyReLU(double value, double alpha = 0.2) => value > 0 ? value : value * alpha;
            public static double SiLU(double value) => value * Sigmoid(value);
            public static double SoftPlus(double value) => Math.Log(1 + Math.Exp(value));
            public static double ELU(double value) => value < 0? Math.Exp(value) - 1 : value;
            public static void SoftMax(double[] values)
            {
                double exp_sum = 0;
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = Math.Exp(values[i]);
                    exp_sum += values[i];
                }

                for (int i = 0; i < values.Length; i++)
                {
                    values[i] /= exp_sum;
                }
            }
            public static void OneHot(double[] values)
            {
                int index = ArgMax(values);
                for (int i = 0; i < values.Length; i++)
                {
                    if (i == index)
                        values[i] = 1;
                    else
                        values[i] = 0;
                }
            }
        }
        public readonly struct Derivative
        {
            public static double DeriveValue(double value, ActivationType activationFunction)
            {
                switch (activationFunction)
                {
                    case ActivationType.Tanh:
                        return TanH(value);
                    case ActivationType.Sigmoid:
                        return Sigmoid(value);
                    case ActivationType.Relu:
                        return ReLU(value);
                    case ActivationType.LeakyRelu:
                        return LeakyReLU(value);
                    case ActivationType.Silu:
                        return SiLU(value);
                    case ActivationType.ELU:
                        return ELU(value);
                    case ActivationType.SoftPlus:
                        return SoftPlus(value);
                    default: // Linear
                        return 1;
                }
            }

            public static double TanH(double  value)
            {
                double e2 = Math.Exp(2 * value);
                double tanh = (e2 - 1) / (e2 + 1);
                return 1 - tanh * tanh;
                
            }
            public static double Sigmoid(double value)
            {
                double act = Activation.Sigmoid(value);
                return act * (1 - act);
            }
            public static double ReLU(double value) => value > 0 ? 1 : 0;
            public static double LeakyReLU(double value, double alpha = 0.2) => value > 0 ? 1 : alpha;
            public static double SiLU(double value)
            {
                double sigm = Activation.Sigmoid(value);
                return value * sigm * (1 - sigm) + sigm;
            }
            public static double ELU(double value) => value < 0 ? Math.Exp(value) : 1;
            public static double SoftPlus(double value) => Activation.Sigmoid(value);
            public static void SoftMax(double[] values)
            {
                double exp_sum = 0;
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = Math.Exp(values[i]);
                    exp_sum += values[i];
                }

                double squared_sum = exp_sum * exp_sum;

                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = (values[i] * exp_sum - values[i] * values[i]) / squared_sum;
                }
            }
        }
        public readonly struct Error
        {
            public static double MeanAbsolute(double prediction, double label)
            {
                return Math.Abs(prediction - label);
            }
            public static double MeanSquare(double prediction, double label)
            {
                return 0.5 * (prediction - label) * (prediction - label);
            }
            public static double CrossEntropy(double prediction, double label)
            {
                double err = -label * Math.Log(prediction);
                return double.IsNaN(err) ? 0 : err;
            }
        }
        public readonly struct Loss
        {
            public static double AbsoluteDerivative(double prediction, double label)
            {
                return prediction - label > 0 ? 1 : -1;
            }
            public static double MeanSquareDerivative(double prediction, double label)
            {
                return prediction - label;
            }
            public static double CrossEntropyDerivative(double prediction, double label)
            {
                return (-prediction + label) / (prediction * (prediction - 1) + 1e-10);
            }
        }

        public readonly struct Image
        {
            public readonly struct Convolution
            {
                static public void Pad(ref float[,] image, PaddingType paddingType)
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
                static public void Filter(ref float[,] image, float[,] kernel)
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
                static public void Normalize01(ref float[,] image)
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
                static public void Pool(ref float[,] image, PoolType poolType)
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
            }
            public readonly struct Kernels
            {
                public static float[,] GetKernel(KernelType kernel)
                {
                    switch(kernel)
                    {
                        case KernelType.Edge_3x3:
                            return kernel3x3_edge;
                        case KernelType.Blur_3x3:
                            return kernel3x3_blur;
                        case KernelType.Emboss_3x3:
                            return kernel3x3_emboss;
                        case KernelType.Laplac_3x3:
                            return kernel3x3_laplacian;
                        case KernelType.MoBlur_3x3:
                            return kernel3x3_mo_blur;
                        case KernelType.Gabor_3x3:
                            return kernel3x3_gabor;
                        case KernelType.Mean_3x3:
                            return kernel3x3_mean;
                        default:
                            throw new Exception("Unhandled kernel type.");

                    }
                }

                internal static float[,] kernel3x3_edge = new float[3, 3]
                {
                    { -1, 0, 1 },
                    { -2, 0, 2 },
                    { -1, 0, 1 }
                };
                internal static float[,] kernel3x3_blur = new float[3, 3]
                {
                    { 1, 2, 1 },
                    { 2, 4, 2 },
                    { 1, 2, 1 }
                };
                internal static float[,] kernel3x3_emboss = new float[3, 3]
                {
                    { -2, -1, 0 },
                    { -1,  1, 1 },
                    {  0,  1, 2 }
                };
                internal static float[,] kernel3x3_laplacian = new float[3, 3]
                {
                    { -1, -1, -1 },
                    { -1,  8, -1 },
                    { -1, -1, -1 }
                };
                internal static float[,] kernel3x3_mo_blur = new float[3, 3]
                {
                    { 1, 0, 0 },
                    { 0, 1, 0 },
                    { 0, 0, 1 }
                };
                internal static float[,] kernel3x3_gabor = new float[3, 3]
                {
                    { -0.087F, -0.341F, -0.087F },
                    { -0.341F,  1.765F, -0.341F },
                    { -0.087F, -0.341F, -0.087F }
                };
                internal static float[,] kernel3x3_mean = new float[3, 3]
                {
                    {0.11111111f, 0.11111111f, 0.11111111f },
                    {0.11111111f, 0.11111111f, 0.11111111f },
                    {0.11111111f, 0.11111111f, 0.11111111f }
                };
            }
           
        }

    }
}