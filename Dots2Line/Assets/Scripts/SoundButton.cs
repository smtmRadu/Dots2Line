using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class SoundButton : MonoBehaviour
{
    public Image thisButtonImage;
    public Sprite image1;
    public Sprite image2;

    public AudioSource audioSource;
    public void ChangeSoundState()
    {
        thisButtonImage.sprite = thisButtonImage.sprite == image1 ? image2 : image1;
        if (audioSource.mute == false)
            audioSource.mute = true;
        else
            audioSource.mute = false;
    }
}
