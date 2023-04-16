using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.UI;
using NeuroForge;

public class Drawer : MonoBehaviour
{
    public Image mainImage;
    public TMPro.TMP_Text predictionText;


    public ConvolutionalNeuralNetwork network;
    public double[] predictions;

    public bool debugImage = false;

    [Range(0f, 1f)] public float pencilStrength = .5f;
    [Min(1f)] public float pencilRadius = 1f;
    private void Awake()
    {
    }
    private void Update()
    {
        Draw();

    }
    private void FixedUpdate()
    {
        Predict();
        if(debugImage) 
            DebugImage();
    }

    private void Draw()
    {
        if (Input.touchCount == 0)
            return;

        Touch touch = Input.GetTouch(0);

        float xPOS = touch.position.x;
        float yPOS = touch.position.y;

        if (!IsInsideFrame_ThenNormalize(ref xPOS, ref yPOS))
            return;

        DrawWithBrushOn((int)xPOS, (int)yPOS);
    }

    private void DrawWithBrushOn(int x, int y)
    {
        Texture2D texture = mainImage.sprite.texture;
        for (int i = (int)(x - pencilRadius); i <= (int)(x + pencilRadius); i++)
        {
            for (int j = (int)(y - pencilRadius); j <= (int)(y + pencilRadius); j++)
            {
                var distance = Vector2.Distance(new Vector2(x, y), new Vector2(i, j));

                if ( distance > pencilRadius)
                    continue;

                var pix = texture.GetPixel(i, j);
                float plusInk = pencilRadius/distance * pencilStrength;
                pix.r += plusInk;
                pix.g += plusInk;
                pix.b += plusInk;
                texture.SetPixel(i, j, pix);

            }
        }
        mainImage.sprite.texture.Apply();

    }
    
    public void Clear()
    {
        Color[] pixels = mainImage.sprite.texture.GetPixels();
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = Color.black;
        }
        mainImage.sprite.texture.SetPixels(pixels);
        mainImage.sprite.texture.Apply();
    }
    private bool IsInsideFrame_ThenNormalize(ref float xPOS, ref float yPOS)
    {
        int width = Screen.width;
        int height = Screen.height;


        // Check y is withing 25%-75%
        if (yPOS < 25f / 100f * height || yPOS > 75f / 100f * height)
            return false;

        // From here is ok

        // Normalize the height to 0 in right up corner
        yPOS -= 25f / 100f * height;

        // Normalize x and y within 0-28 range
        xPOS = xPOS / (float)width * 28f;

        // Same for the height
        yPOS = yPOS / (float)(height * .5f) * 28f;

        return true;
    }




    const string confidence_75_100 = "Definetly a ";
    const string confidence_50_75 = "Probably a ";
    const string confidence_25_50 = "Maybe a ";
    const string confidence_0_25 = "Hard to say, can be anything...";
    private void Predict()
    {
        float[] inputs = mainImage.sprite.texture.GetPixels().Select(x => x.grayscale).ToArray();
        predictions = network.Forward(ToMatrix(inputs));



        /// after prediction
        int predicition = Functions.ArgMax(predictions);

        float confidence01 = (float)predictions.Max();


        string finalSTRING = string.Empty;
        switch (predictions.Max())
        {
            case >= .75f:
                finalSTRING = confidence_75_100 + predicition;
                break;
            case >= 0.5f:
                finalSTRING = confidence_50_75 + predicition;
                break;
            case >= 0.25f:
                finalSTRING = confidence_25_50 + predicition;
                break;
            default:
                finalSTRING = confidence_0_25;
                break;
        }
        predictionText.text = finalSTRING;

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
    private void DebugImage()
    {
        double[] pixels = mainImage.sprite.texture.GetPixels().Select(x => (double)x.grayscale).ToArray();
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
}