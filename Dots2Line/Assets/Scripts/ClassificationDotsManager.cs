using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// NOTE: Whenever you want to add a new class of dots, please add the following:
/// on DotsManager, add new dot prefab
/// on NetworkManager, add new color
/// on canvas, DotTypeButton, add the sprite
/// </summary>
public class ClassificationDotsManager : MonoBehaviour
{

    public ClassificationNetworkManager NetworkManager;
    public WarningsPrinter warnPrinter;
    public List<GameObject> dotPrefabs = new List<GameObject>();
    public int DOT_IN_USE_INDEX = 0;

    [Space]
    public float dotsZglobalPosition = 5f;
    public int minDots = 10;
    public int maxDots = 100;
    public float placeRate = 1.0f;
    private float placeTimeLeft = 0f;

    private List<ColoredDot> trainDots = new List<ColoredDot>();

    public int dots_on_the_map = 0;

    void Update()
    {
        placeTimeLeft -= Time.deltaTime;
        DrawDots();

    }

    public void ChangeDotType()
    {
        if (DOT_IN_USE_INDEX == dotPrefabs.Count - 1)
            DOT_IN_USE_INDEX = 0;
        else
            DOT_IN_USE_INDEX++;
        
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
        if (Physics.Raycast(ray, out hit))
        {
            if (hit.collider.CompareTag("Grid"))
            {
                Vector3 newPosition = Camera.main.ScreenToWorldPoint(firstTouch.position);
                newPosition.z = dotsZglobalPosition;
                GameObject newDot = Instantiate(dotPrefabs[DOT_IN_USE_INDEX], newPosition, Quaternion.identity, this.transform);
                int newDotTYPE = DOT_IN_USE_INDEX;
                trainDots.Add(new ColoredDot(newDot, newDotTYPE));
                dots_on_the_map++;
            }
        }



    }

    public void ResetSimulation()
    {
        foreach (var dot in trainDots)
        {
            Destroy(dot.go);
        }
        trainDots.Clear();
        dots_on_the_map = 0;

        NetworkManager.neuralNetwork = null;
        NetworkManager.error = 0;
        NetworkManager.state = NetManagerState.Start;
    }
    public void PauseSimulation()
    {
        if(NetworkManager.state == NetManagerState.Running)
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

        // Find how many types of dots there are -- here was a bug regarding to type, labels in network and other.. 
        int count = trainDots.Select(x => x.type).Max() + 1;
        // detailed why here is not Distinct().Count() -> because dot1 & dot2 are used and not dot0, and labels length will be [0,1]. When you acces labels[type.dot] and type was 2, index outside of the array
        if (count == 1)
        {
            warnPrinter.Print("You drew a single class of dots!");
            return;
        }

        
        NetworkManager.StartLearn(trainDots, count);
    }

}

public class ColoredDot
{
    public GameObject go;
    public float x = 0;
    public float y = 0;
    public int type = 0; // 0, 1, 2... 
    public ColoredDot(GameObject dot, int tp)
    {
        go = dot;
        if (dot == null)
            return;
        x = dot.transform.position.x;
        y = dot.transform.position.y;
        this.type = tp;
    }
    public object Clone()
    {
        ColoredDot clone = new ColoredDot(null, 0);
        clone.go = go;
        clone.x = x;
        clone.y = y;
        clone.type = type;
        return clone;
    }
}