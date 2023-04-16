using System.Collections;
using System.Collections.Generic;
using System.Xml.Serialization;
using UnityEngine;

public class WarningsPrinter : MonoBehaviour
{

    public TMPro.TMP_Text text;
    public void Print(string message, float fadeT = 1f)
    {
        text.color = Color.white;
        text.text = message;
        StartCoroutine(Fade(fadeT));
    }

    IEnumerator Fade(float fadeTime)
    {
        yield return new WaitForSeconds(fadeTime);
        float elapsedTime = 0f;
        while (elapsedTime < fadeTime)
        {
            float alpha = Mathf.Lerp(1f, 0f, elapsedTime / fadeTime);
            text.color = new Color(text.color.r, text.color.g, text.color.b, alpha);
            elapsedTime += Time.deltaTime;
            yield return null;
        }
    }
}
