using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;
using NeuroForge;

public class ClassificationNetworkManager : MonoBehaviour
{
    public ClassificationDotsManager dotsManager;
    public NeuralNetwork neuralNetwork;  
    public WarningsPrinter warningsPrinter;
    private int number_of_outputs = 2;
    [Space]
    public NetManagerState state = NetManagerState.Start;
    public double error;

    public List<Color> predictionColors = new List<Color>()
    { 
            
            Color.blue,
            Color.red,
            Color.green,
    
    };

    [Space]

    [Tooltip("  39x39 : 0.10f\n" +
             "  69x69 : 0.059f\n" +
             "101x101 : 0.0038f\n")]
    public float WHAT_IS_THE_SCALE_TO_RENDER_THE_PREDICTION = 0.059f;

    [Header("To Assign")]
    
    public TMPro.TMP_Dropdown initType;
    public TMPro.TMP_Dropdown activType;
    public Slider learnRate;
    public Slider momentum;
    public Slider regularization;
    public Spinner hidUnits;
    public Spinner layNum;
    public TMPro.TMP_Text toWriteAccuracy;

    public bool parallel = true;
    public bool useGradClipNorm = false;
    public float gradClipNorm = 0.5f;

    private List<ColoredDot> trainDataSet;
    public Image imagePrediction;
    public float FPS = 0f;

    // Because the performance is low, the background is updateting at 25fps
    [Tooltip("frames")] public int UPDATE_ONCE_PER = 5;
    
    public void StartLearn(List<ColoredDot> trainingDataSet, int no_outputs)
    {
        number_of_outputs = no_outputs;
        trainDataSet = trainingDataSet;
        state = NetManagerState.Running;
        
        
    }

    public void Update()
    {
        FPS = 1f / Time.deltaTime;
    }

    private void FixedUpdate()
    {
        // Train the network
        if (state == NetManagerState.Running)
        {
            TrainNetwork();
            double accuracy = (1.0 - error) * 100;
            string acc_string = "Accuracy: " + accuracy.ToString("0.00000") + "%";
            toWriteAccuracy.text = acc_string;
        }
        else
        {
            toWriteAccuracy.text = "";
        }

    }

    void TrainNetwork()
    {
        if (neuralNetwork == null)
        {
            neuralNetwork = new NeuralNetwork(
            2,
            number_of_outputs,
            hidUnits.value,
            layNum.value,
            (ActivationType)activType.value,
            ActivationType.SoftMax,
            LossType.CrossEntropy,
            (InitializationType)initType.value,
            false,
            "classifNetwork");
        }


        //Functions.Shuffle(trainDataSet);
        if(parallel)
        {
            object lock_er = new object();
            System.Threading.Tasks.Parallel.ForEach(trainDataSet, dot =>
            {
                double[] inputs = new double[] { dot.x, dot.y };
                double[] labels = new double[number_of_outputs];
                labels[dot.type] = 1;

                double localError = neuralNetwork.Backward(inputs, labels, true);

                lock(lock_er)
                {
                    error += localError;
                }
            });
            neuralNetwork.OptimStep(learnRate.value, momentum.value, regularization.value);
        }
        else
        {
            foreach (ColoredDot dot in trainDataSet)
            {
                double[] inputs = new double[] { dot.x, dot.y };
                double[] labels = new double[number_of_outputs];
                labels[dot.type] = 1;
                error += neuralNetwork.Backward(inputs, labels, false);
                if (useGradClipNorm)
                    neuralNetwork.GradClipNorm(gradClipNorm);
                neuralNetwork.OptimStep(learnRate.value * 10f, momentum.value, regularization.value); //because this is not SGD, is full batch training, we increase the learn rate
            }
            
        }



        error /= trainDataSet.Count;
        // Test the network

        if (Time.frameCount % UPDATE_ONCE_PER != 0)
            return;

        Color[] pixels = imagePrediction.sprite.texture.GetPixels();
        int length = (int) Mathf.Sqrt(pixels.Length);

        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < length; j++)
            {
                float gridX = j - length / 2;
                float gridY = i - length / 2;

                float firstInput = gridX * WHAT_IS_THE_SCALE_TO_RENDER_THE_PREDICTION;
                float secondInput = gridY * WHAT_IS_THE_SCALE_TO_RENDER_THE_PREDICTION;

                double[] inputs = new double[] { firstInput, secondInput};
                int outp = Functions.ArgMax(neuralNetwork.Forward(inputs));


                pixels[i * length + j] = predictionColors[outp];
            }
        }
        imagePrediction.sprite.texture.SetPixels(pixels);
        imagePrediction.sprite.texture.Apply();
    }

    public void WhiteImagePrediction()
    {
        Color[] pixels = imagePrediction.sprite.texture.GetPixels();

        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = Color.white;
        }
        imagePrediction.sprite.texture.SetPixels(pixels);
        imagePrediction.sprite.texture.Apply();
    }    
}
