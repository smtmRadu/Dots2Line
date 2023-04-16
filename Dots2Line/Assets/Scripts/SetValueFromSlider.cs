using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class SetValueFromSlider : MonoBehaviour
{
    public int decimals = 4;
    public Slider slider;
    public TMP_Text toText;
    private void Update()
    {
        string decimal_format = "0.";
        for (int i = 0; i < decimals; i++)
            decimal_format += "0";
        toText.text = slider.value.ToString(decimal_format);
    }

}
