using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

public class RegressionDotsManager : MonoBehaviour
{
    public RegressionNetworkManager NetworkManager;
    public WarningsPrinter warnPrinter;
    public GameObject dotPrefab;
    public LineRenderer lineRenderer;

    public float dotsZglobalPosition = 5f;
    public int minDots = 10;
    public int maxDots = 100;
    public float placeRate = 1.0f;
    private float placeTimeLeft = 0f;

    [Min(30)] public int noTestDots = 100;

    private List<Dot> trainDots = new List<Dot>();
    private List<Dot> testDots = new List<Dot>();
    public int dots_on_the_map = 0;


    // Update is called once per frame
    void Update()
    {
        placeTimeLeft -= Time.deltaTime;
        DrawDots();
       
    }
    private void FixedUpdate()
    {
        DrawLine();
    }

    void DrawDots()
    {
        if (placeTimeLeft > 0f)
            return;

        if (Input.touchCount == 0)
            return;

        if (NetworkManager.state != NetManagerState.Start)
            return;

        if (dots_on_the_map >= maxDots)
        {
            warnPrinter.Print("Enough dots!");
            return;
        }

        placeTimeLeft = 1f / placeRate;

        Touch firstTouch = Input.GetTouch(0);


        Ray ray = Camera.main.ScreenPointToRay(firstTouch.position);
        RaycastHit hit;
        if(Physics.Raycast(ray, out hit))
        {
            if(hit.collider.CompareTag("Grid"))
            {
                Vector3 newPosition = Camera.main.ScreenToWorldPoint(firstTouch.position);
                newPosition.z = dotsZglobalPosition;
                GameObject newDot = Instantiate(dotPrefab, newPosition, Quaternion.identity, this.transform);
                trainDots.Add(new Dot(newDot));
                dots_on_the_map++;
            }
        }
        
        
        
    }
    void DrawLine()
    {
        List<Vector3> positions = new List<Vector3>();
        foreach (var item in testDots)
        {
            positions.Add(new Vector3(item.x, item.y, dotsZglobalPosition));
        }
        lineRenderer.positionCount = positions.Count;
        lineRenderer.SetPositions(positions.ToArray());


    }
    public void ResetSimulation()
    {
        foreach (var dot in trainDots)
        {
            Destroy(dot.go);
        }
        trainDots.Clear();
        testDots.Clear();
        dots_on_the_map = 0;

        NetworkManager.neuralNetwork = null;
        NetworkManager.error = 0;
        NetworkManager.state = NetManagerState.Start;
    }
    public void PauseSimulation()
    {
        if (NetworkManager.state == NetManagerState.Running)
            NetworkManager.state = NetManagerState.Paused;
    }
    public void StartSimulation()
    {
        if (NetworkManager.state == NetManagerState.Running)
            return;

        if (trainDots.Count < minDots)
        {
            /// Warning message
            warnPrinter.Print("Draw more dots!");
            return;
        }

        // Create the test dots -----------------------------------------------------------------------------
        // Find smallest dot by x
        // Find largest dot by y

        Dot left_dot = null;
        Dot right_dot = null;
        foreach (var item in trainDots)
        {
            if (left_dot == null || item.x < left_dot.x)
                left_dot = item;
            if(right_dot == null || item.x > right_dot.x)
                right_dot = item;
        }
        float step_on_x = (right_dot.x - left_dot.x)/noTestDots;
        float current_step_on_x = left_dot.x;
        for (int i = 0; i < noTestDots; i++)
        {
            Dot newDot = new Dot(null);
            newDot.y = 0;
            newDot.x = current_step_on_x;
            current_step_on_x += step_on_x;
            testDots.Add(newDot);
        }
        testDots.Sort((dot1, dot2) => dot1.x.CompareTo(dot2.x));

        
        NetworkManager.Learn(trainDots, testDots);
    }


    public void NormalizeTrainingDataSet()
    {
        //  it does not really need normalization
    }
    public void DenormalizeTestDataSet()
    {
        // it does not really need normalization.. so no denormalization of tests
    }


    
}

public class Dot : ICloneable
{
    public GameObject go;
    public float x = 0;
    public float y = 0;

    public Dot(GameObject dot)
    {
        go = dot;
        if (dot == null)
            return;
        x = dot.transform.position.x;
        y = dot.transform.position.y;
    }
     public object Clone()
     {
        Dot clone = new Dot(null);
        clone.go = go;
        clone.x = x;
        clone.y = y;
        return clone;
     }

}