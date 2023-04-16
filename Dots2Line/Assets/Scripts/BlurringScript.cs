using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BlurringScript : MonoBehaviour
{
    public Material blurMaterial;
    public float blurSize = 2.0f;

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        // Apply the blur effect
        Graphics.Blit(source, destination, blurMaterial);
    }

}
