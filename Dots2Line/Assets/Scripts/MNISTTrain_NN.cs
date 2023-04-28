using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using static UnityEngine.Mesh;
using NeuroForge;
using System.Text;

public class MNISTTrain_NN : MonoBehaviour
{
    public ConvolutionalNeuralNetwork network;
    public int miniBatchSize_X10 = 64;
    List<(int, float[])> trainData;
    List<(int, float[])> testData;
    public bool parallel = true;
    public float FPS = 0;
    [Space]
    public int hid_units = 128;
    public int lay_number = 2;
    [Range(0f,0.01f)]public float learnRate = 0.0003f;
    public float momentum = 0.9f;
    public float regularization = 0.00005f;
    public int convolLvl = 3;

    public string trainAcc;
    public string testAcc;
    public string digitTestAcc;

    public bool printWith01 = false;

    private void Awake()
    {
        Application.runInBackground = true;
        if (network == null)
        {
            network = new ConvolutionalNeuralNetwork(28, 28, 10, convolLvl, hid_units, lay_number, true, "CNN");
            //network = new NeuralNetwork(784, 10, hid_units, lay_number, ActivationType.Relu, ActivationType.SoftMax, LossType.CrossEntropy, InitializationType.He ,true, "MNIST");
        }
    }

    public void Update()
    {
        GenerateTrainData();
        GenerateTestData();

        Train();
        Test();
        //network.Save();

        FPS = 1f/Time.deltaTime;
    }
    void GenerateTrainData()
    {
        trainData = new List<(int, float[])>();

        string trainPath = "C:\\Users\\X\\Desktop\\TRAIN\\";
        for (int i = 0; i < 10; i++)
        {
            trainPath += i;
            string[] imagesPaths = Directory.GetFiles(trainPath, "*.jpg", SearchOption.TopDirectoryOnly);

            for (int j = 0; j < miniBatchSize_X10; j++)
            {
                float[] imgPix = LoadTexture(Functions.RandomIn(imagesPaths)).GetPixels().Select(x => x.grayscale).ToArray();

                trainData.Add((i, imgPix));
            }
            trainPath = trainPath.Substring(0, trainPath.Length - 1);
        }
    } 
    void GenerateTestData()
    {
        testData = new List<(int, float[])>();


        string testPath = "C:\\Users\\X\\Desktop\\TEST\\";
        for (int i = 0; i < 10; i++)
        {
            testPath += i;
            string[] imagesPaths = Directory.GetFiles(testPath, "*.jpg", SearchOption.TopDirectoryOnly);

            for (int j = 0; j < miniBatchSize_X10/2; j++)
            {
                float[] imgPix = LoadTexture(imagesPaths[j]).GetPixels().Select(x => x.grayscale).ToArray();
                testData.Add((i, imgPix));
            }
            testPath = testPath.Substring(0, testPath.Length - 1);
        }
    }

    public static float[,] AugmentData(float[,] img)
    {
        img = RotateMatrix(img, Functions.RandomRange(-15, 16));
        img = AddGaussianNoise(img, 0, (float)Functions.RandomRange(0.2f, 1.2f));
        img = Zoom(img, (float)Functions.RandomRange(0.5f, 3f));
        return img;
    }
    static float[,] RotateMatrix(float[,] matrix, float angleDegrees)
    {
        float angleRadians = angleDegrees * Mathf.PI / 180f;
        float cosTheta = Mathf.Cos(angleRadians);
        float sinTheta = Mathf.Sin(angleRadians);
        int centerX = matrix.GetLength(0) / 2;
        int centerY = matrix.GetLength(1) / 2;

        float[,] result = new float[matrix.GetLength(0), matrix.GetLength(1)];

        for (int x = 0; x < matrix.GetLength(0); x++)
        {
            for (int y = 0; y < matrix.GetLength(1); y++)
            {
                int rotatedX = (int)(cosTheta * (x - centerX) + sinTheta * (y - centerY) + centerX);
                int rotatedY = (int)(-sinTheta * (x - centerX) + cosTheta * (y - centerY) + centerY);

                if (rotatedX >= 0 && rotatedX < matrix.GetLength(0) && rotatedY >= 0 && rotatedY < matrix.GetLength(1))
                {
                    result[x, y] = matrix[rotatedX, rotatedY];
                }
            }
        }

        return result;
    }
    static float[,] AddGaussianNoise(float[,] matrix, float mean, float stdDev)
    {
        float[,] noisyMatrix = new float[matrix.GetLength(0), matrix.GetLength(1)];

        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            for (int j = 0; j < matrix.GetLength(1); j++)
            {
                float noise = FunctionsF.RandomGaussian(mean, stdDev);
                noisyMatrix[i, j] = matrix[i, j] + noise;
            }
        }

        return noisyMatrix;
    }
    static float[,] Zoom(float[,] input, float zoomFactor)
    {
        int inputWidth = input.GetLength(0);
        int inputHeight = input.GetLength(1);

        int outputWidth = Mathf.RoundToInt(inputWidth * zoomFactor);
        int outputHeight = Mathf.RoundToInt(inputHeight * zoomFactor);

        float[,] output = new float[outputWidth, outputHeight];

        for (int y = 0; y < outputHeight; y++)
        {
            for (int x = 0; x < outputWidth; x++)
            {
                float u = (float)x / (float)outputWidth * (float)inputWidth;
                float v = (float)y / (float)outputHeight * (float)inputHeight;

                int x0 = Mathf.FloorToInt(u);
                int y0 = Mathf.FloorToInt(v);
                int x1 = Mathf.Min(x0 + 1, inputWidth - 1);
                int y1 = Mathf.Min(y0 + 1, inputHeight - 1);

                float tx = u - (float)x0;
                float ty = v - (float)y0;

                float w00 = (1.0f - tx) * (1.0f - ty);
                float w01 = (1.0f - tx) * ty;
                float w10 = tx * (1.0f - ty);
                float w11 = tx * ty;

                output[x, y] = w00 * input[x0, y0] + w01 * input[x0, y1] + w10 * input[x1, y0] + w11 * input[x1, y1];
            }
        }

        return output;
    }



    float[,] ToMatrix(float[] flat)
    {
        float[,] mat = new float[28, 28];
        for (int x = 0; x < 28; x++)
        {
            for (int y = 0; y < 28; y++)
            {
                mat[x, y] = flat[x * 28 + y];
            }
        }
        return mat;
    }
    void Train()
    {
        double err = 0.0;
        int count = 0;
        object loc_em = new object();

        if (parallel)
        {
            System.Threading.Tasks.Parallel.ForEach(trainData, digit =>
            {
                //double[] labels = new double[10]; labels[digit.Item1] = 1;
                double errX = network.Backward(ToMatrix(digit.Item2), digit.Item1, true);

                lock (loc_em)
                {
                    err += errX;
                    count++;
                }               
            });
            
        }
        else
        {
           //
        }

        network.OptimStep(learnRate, momentum, regularization);
        trainAcc = ((1.0 - err / count) * 100).ToString("0.000") + "%";
    }
    void Test()
    {
        double err = 0.0;
        int correct = 0;
        int wrong = 0;
        foreach (var digit in testData)
        {
            double[] output = network.Forward(ToMatrix(digit.Item2));
            int predict = Functions.ArgMax(output);

            for (int i = 0; i < 10; i++)
            {
                err += Functions.Error.MeanSquare(output[i], digit.Item1 == i ? 1 : 0);
            }

            if (predict == digit.Item1)
                correct++;
            else
                wrong++;
        }

        testAcc = ((1.0 - err / (correct + wrong)) * 100).ToString("0.000") + "%";
        digitTestAcc = ((correct/(float)(correct + wrong)) * 100).ToString("0.000") + "%";
    }
    private void DebugImage()
    {
        Texture2D here = LoadTexture("C:\\Users\\X\\Desktop\\TRAIN\\3\\86.jpg");
        double[] pixels = here.GetPixels().Select(x => (double)x.grayscale).ToArray();
        for (int i = 0; i < pixels.Length; i++)
        {
            if (pixels[i] < 0.5)
                pixels[i] = 0;
            else
                pixels[i] = 1;
        }

        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < 28; i++)
        {

            for (int j = 0; j < 28; j++)
            {
                stringBuilder.Append(pixels[i * 28 + j]);
            }
            stringBuilder.Append('\n');
        }

        Debug.Log(stringBuilder.ToString());
    }
    private Texture2D LoadTexture(string filePath)
    {
        Texture2D tex = null;
        byte[] fileData;

        if (File.Exists(filePath))
        {
            fileData = File.ReadAllBytes(filePath);
            tex = new Texture2D(28, 28);
            tex.LoadImage(fileData);
        }
        return tex;
    }
}
