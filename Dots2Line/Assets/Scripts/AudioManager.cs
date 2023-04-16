using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AudioManager : MonoBehaviour
{
    public AudioSource mainMenuMusic;


    public void Play_MainMenuMusic()
    {
        mainMenuMusic.Play();
    }
    public void Stop_MainMenuMusic()
    {
        mainMenuMusic.Stop();
    }
}
