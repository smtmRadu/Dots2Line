using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using NeuroForge;

public class RegressionNetworkManager : MonoBehaviour
{
    /// <summary>
    /// /// ALL DATA IS BY DEFAULT NORMALIZED, the values are in small ranges [-1,1] on X Axis, [-1.5, 1.5] on Y axis, so they are good to go
    /// </summary>
    public RegressionDotsManager dotsManager;
    public NeuralNetwork neuralNetwork;
    public NetManagerState state = NetManagerState.Start;
    public WarningsPrinter warningsPrinter;
    public double error;

    [Header("To Assign")]
    public TMPro.TMP_Dropdown initType;
    public TMPro.TMP_Dropdown activType;
    public Slider learnRate;
    public Slider momentum;
    public Slider regularization;
    public Spinner hidUnits;
    public Spinner layNum;
    public TMPro.TMP_Text toWriteAccuracy;

    public bool useGradClipNorm = false;
    public float gradClipNorm = 0.5f;

    private List<Dot> trainDataSet;
    private List<Dot> testDataSet;

    public void Learn(List<Dot> trainingDataSet, List<Dot> testDataSet)
    {
        state = NetManagerState.Running;
        trainDataSet = trainingDataSet;
        this.testDataSet = testDataSet;
       
    }

    private void FixedUpdate()
    {
        // Train the network
        if(state == NetManagerState.Running)
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
            1,
            1,
            hidUnits.value,
            layNum.value,
            (ActivationType)activType.value,
            ActivationType.Linear,
            LossType.MeanSquare,
            (InitializationType)initType.value,
            false,
            "runNetwork");
        }


        Functions.Shuffle(trainDataSet);
        foreach (Dot dot in trainDataSet)
        {
            double[] inputs = new double[] { dot.x };
            double[] labels = new double[] { dot.y };
            error += neuralNetwork.Backward(inputs, labels);
            if (useGradClipNorm)
                neuralNetwork.GradClipNorm(gradClipNorm);
            neuralNetwork.OptimStep(learnRate.value, momentum.value, regularization.value);
        }
        error /= trainDataSet.Count;
        // Test the network
        for (int i = 0; i < testDataSet.Count; i++)
        {
            double[] inputs = new double[] { testDataSet[i].x };
            double[] output = neuralNetwork.Forward(inputs);
            testDataSet[i].y = (float)output[0];
        }


    }
}
