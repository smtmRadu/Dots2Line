using NeuroForge;
using UnityEngine;
using UnityEngine.UI;
public class FunctionsMenu : MonoBehaviour
{
    [SerializeField] RegressionDotsManager m_RegressionDotsManager;
    [SerializeField] Button m_ResetButton;
    [SerializeField] int how_many_dots = 100;
    [SerializeField] float stdDev = 8f;
    [SerializeField] float scale = 0.3f;
    [SerializeField] Toggle m_open_links_on_click;
    // Math
    public void DrawDots_Sine()
    {
        m_ResetButton.onClick.Invoke();
    
        for (int i = 0; i < how_many_dots; i++)
        {
            float newX = FunctionsF.RandomGaussian(0, stdDev/ 2);
            float newY = Mathf.Sin(newX);
            newX *= scale;
            newY *= scale;
            Vector3 newPosition = new Vector3(newX, newY, 9);
            GameObject newDot = Instantiate(m_RegressionDotsManager.dotPrefab, newPosition, Quaternion.identity, m_RegressionDotsManager.transform);
            m_RegressionDotsManager.trainDots.Add(new Dot(newDot));
            m_RegressionDotsManager.dots_on_the_map++;
        }
    }
    public void DrawDots_Square()
    {
        m_ResetButton.onClick.Invoke();

        for (int i = 0; i < how_many_dots; i++)
        {
            float newX = FunctionsF.RandomGaussian(0, stdDev);
            float newY = newX * newX;
            newX *= scale;
            newY *= scale;
            Vector3 newPosition = new Vector3(newX, newY, 9);
            GameObject newDot = Instantiate(m_RegressionDotsManager.dotPrefab, newPosition, Quaternion.identity, m_RegressionDotsManager.transform);
            m_RegressionDotsManager.trainDots.Add(new Dot(newDot));
            m_RegressionDotsManager.dots_on_the_map++;
        }
    }
    public void DrawDots_Cube()
    {
        m_ResetButton.onClick.Invoke();

        for (int i = 0; i < how_many_dots; i++)
        {
            float newX = FunctionsF.RandomGaussian(0, stdDev/7);
            float newY = newX * newX * newX;
            newX *= scale;
            newY *= scale;
            Vector3 newPosition = new Vector3(newX, newY, 9);
            GameObject newDot = Instantiate(m_RegressionDotsManager.dotPrefab, newPosition, Quaternion.identity, m_RegressionDotsManager.transform);
            m_RegressionDotsManager.trainDots.Add(new Dot(newDot));
            m_RegressionDotsManager.dots_on_the_map++;
        }
    }
    public void DrawDots_Cos()
    {
        m_ResetButton.onClick.Invoke();

        for (int i = 0; i < how_many_dots; i++)
        {
            float newX = FunctionsF.RandomGaussian(0, stdDev);
            float newY = Mathf.Cos(newX);
            newX *= scale;
            newY *= scale;
            Vector3 newPosition = new Vector3(newX, newY, 9);
            GameObject newDot = Instantiate(m_RegressionDotsManager.dotPrefab, newPosition, Quaternion.identity, m_RegressionDotsManager.transform);
            m_RegressionDotsManager.trainDots.Add(new Dot(newDot));
            m_RegressionDotsManager.dots_on_the_map++;
        }
    }
    public void DrawDots_Inverse()
    {
        m_ResetButton.onClick.Invoke();

        for (int i = 0; i < how_many_dots; i++)
        {
            float newX = FunctionsF.RandomGaussian(0, stdDev);
            float newY = 1f / newX;
            newX *= scale;
            newY *= scale;
            Vector3 newPosition = new Vector3(newX, newY, 9);
            GameObject newDot = Instantiate(m_RegressionDotsManager.dotPrefab, newPosition, Quaternion.identity, m_RegressionDotsManager.transform);
            m_RegressionDotsManager.trainDots.Add(new Dot(newDot));
            m_RegressionDotsManager.dots_on_the_map++;
        }
    }
    public void DrawDots_Int()
    {
        m_ResetButton.onClick.Invoke();

        for (int i = 0; i < how_many_dots * 1.5f; i++)
        {
            float newX = FunctionsF.RandomGaussian(0, stdDev);
            float newY = Mathf.Floor(newX);
            newX *= scale;
            newY *= scale;
            Vector3 newPosition = new Vector3(newX, newY, 9);
            GameObject newDot = Instantiate(m_RegressionDotsManager.dotPrefab, newPosition, Quaternion.identity, m_RegressionDotsManager.transform);
            m_RegressionDotsManager.trainDots.Add(new Dot(newDot));
            m_RegressionDotsManager.dots_on_the_map++;
        }
    }

    //Activation
    public void DrawDots_Sigmoid()
    {
        m_ResetButton.onClick.Invoke();

        for (int i = 0; i < how_many_dots; i++)
        {
            float newX = FunctionsF.RandomGaussian(0, stdDev);
            float newY = (float)Functions.Activation.Sigmoid(newX);
            newX *= scale;
            newY *= scale;
            Vector3 newPosition = new Vector3(newX, newY, 9);
            GameObject newDot = Instantiate(m_RegressionDotsManager.dotPrefab, newPosition, Quaternion.identity, m_RegressionDotsManager.transform);
            m_RegressionDotsManager.trainDots.Add(new Dot(newDot));
            m_RegressionDotsManager.dots_on_the_map++;
        }
    }
    public void DrawDots_ReLU()
    {
        m_ResetButton.onClick.Invoke();

        for (int i = 0; i < how_many_dots; i++)
        {
            float newX = FunctionsF.RandomGaussian(0, stdDev);
            float newY = (float)Functions.Activation.ReLU(newX);
            newX *= scale;
            newY *= scale;
            Vector3 newPosition = new Vector3(newX, newY, 9);
            GameObject newDot = Instantiate(m_RegressionDotsManager.dotPrefab, newPosition, Quaternion.identity, m_RegressionDotsManager.transform);
            m_RegressionDotsManager.trainDots.Add(new Dot(newDot));
            m_RegressionDotsManager.dots_on_the_map++;
        }
    }
    public void DrawDots_LeakyReLU()
    {
        m_ResetButton.onClick.Invoke();

        for (int i = 0; i < how_many_dots; i++)
        {
            float newX = FunctionsF.RandomGaussian(0, stdDev);
            float newY = (float)Functions.Activation.LeakyReLU(newX);
            newX *= scale;
            newY *= scale;
            Vector3 newPosition = new Vector3(newX, newY, 9);
            GameObject newDot = Instantiate(m_RegressionDotsManager.dotPrefab, newPosition, Quaternion.identity, m_RegressionDotsManager.transform);
            m_RegressionDotsManager.trainDots.Add(new Dot(newDot));
            m_RegressionDotsManager.dots_on_the_map++;
        }
    }
    public void DrawDots_SiLU()
    {
        m_ResetButton.onClick.Invoke();

        for (int i = 0; i < how_many_dots; i++)
        {
            float newX = FunctionsF.RandomGaussian(0, stdDev);
            float newY = (float)Functions.Activation.SiLU(newX);
            newX *= scale;
            newY *= scale;
            Vector3 newPosition = new Vector3(newX, newY, 9);
            GameObject newDot = Instantiate(m_RegressionDotsManager.dotPrefab, newPosition, Quaternion.identity, m_RegressionDotsManager.transform);
            m_RegressionDotsManager.trainDots.Add(new Dot(newDot));
            m_RegressionDotsManager.dots_on_the_map++;
        }
    }
    public void DrawDots_SoftPlus()
    {
        m_ResetButton.onClick.Invoke();

        for (int i = 0; i < how_many_dots; i++)
        {
            float newX = FunctionsF.RandomGaussian(0, stdDev);
            float newY = (float)Functions.Activation.SoftPlus(newX);
            newX *= scale;
            newY *= scale;
            Vector3 newPosition = new Vector3(newX, newY, 9);
            GameObject newDot = Instantiate(m_RegressionDotsManager.dotPrefab, newPosition, Quaternion.identity, m_RegressionDotsManager.transform);
            m_RegressionDotsManager.trainDots.Add(new Dot(newDot));
            m_RegressionDotsManager.dots_on_the_map++;
        }
    }
    public void DrawDots_TanH()
    {
        m_ResetButton.onClick.Invoke();

        for (int i = 0; i < how_many_dots; i++)
        {
            float newX = FunctionsF.RandomGaussian(0, stdDev);
            float newY = (float)Functions.Activation.TanH(newX);
            newX *= scale;
            newY *= scale;
            Vector3 newPosition = new Vector3(newX, newY, 9);
            GameObject newDot = Instantiate(m_RegressionDotsManager.dotPrefab, newPosition, Quaternion.identity, m_RegressionDotsManager.transform);
            m_RegressionDotsManager.trainDots.Add(new Dot(newDot));
            m_RegressionDotsManager.dots_on_the_map++;
        }
    }
    // URLs for all
    public void OpenURL_SineCosine()
    {
        if (!m_open_links_on_click.isOn)
            return;

        Application.OpenURL("https://en.wikipedia.org/wiki/Sine_and_cosine");
    }
    public void OpenURL_Cube()
    {
        if (!m_open_links_on_click.isOn)
            return;
        Application.OpenURL("https://www.bing.com/search?q=cube+function+wikipedia&qs=n&form=QBRE&sp=-1&ghc=1&lq=0&pq=cube+function+wikipedi&sc=9-22&sk=&cvid=F42B46D50AAB4AC094A572BA8474699C&ghsh=0&ghacc=0&ghpl=");
    }
    public void OpenURL_Square()
    {
        if (!m_open_links_on_click.isOn)
            return;
        Application.OpenURL("https://en.wikipedia.org/wiki/Square_(algebra)");
    }
    public void OpenURL_Int()
    {
        if (!m_open_links_on_click.isOn)
            return;
        Application.OpenURL("https://en.wikipedia.org/wiki/Floor_and_ceiling_functions");
    }
    public void OpenURL_Inverse()
    {
        if (!m_open_links_on_click.isOn)
            return;
        Application.OpenURL("https://en.wikipedia.org/wiki/Inverse_function");
    }




    public void OpenURL_Sigmoid()
    {
        if (!m_open_links_on_click.isOn)
            return;
        Application.OpenURL("https://en.wikipedia.org/wiki/Sigmoid_function");
    }
    public void OpenURL_TanH()
    {
        if (!m_open_links_on_click.isOn)
            return;
        Application.OpenURL("https://en.wikipedia.org/wiki/Hyperbolic_functions");
    }
    public void OpenURL_Rectifier()
    {
        if (!m_open_links_on_click.isOn)
            return;
        Application.OpenURL("https://en.wikipedia.org/wiki/Rectifier_(neural_networks)");
    }
}