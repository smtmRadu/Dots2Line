using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class Spinner : MonoBehaviour
{
    public int value = 0;
    public Vector2 range = new Vector2(0,100);

    [Space]
    public TMP_Text valueText;
    public Button leftButton;
    public Button rightButton;

    public void Decrement()
    {
        value = Mathf.Clamp(value - 1, (int)range.x, (int)range.y);
        UpdateText();
    }
    public void Increment() {
        value = Mathf.Clamp(value + 1, (int)range.x, (int)range.y);
        UpdateText();
    }
    public void UpdateText()
    {
        valueText.text = value.ToString();
    }

}
