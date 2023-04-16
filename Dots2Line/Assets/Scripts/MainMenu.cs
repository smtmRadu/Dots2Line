using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using NeuroForge;

public class MainMenu : MonoBehaviour
{
    // extra time is .33f;
    public float load_time = 1.5f;
    public float elapsed_time = 0f;
    public TMPro.TMP_Text text;
    public Slider loadSlide;

    public GameObject mainMenuCanvas;
    public GameObject loadingCanvas;
    public float BGMusicDecreaseTime = 1.88f;
   

    List<string> loadingTips = new List<string>()
    {
        "Loading...\nTIP: Shoot at the enemy players to kill them.",
        "Loading...\nTIP: The chances of hitting your target goes up dramatically when aiming at them.",
        "Loading...\nTIP: If you're in trouble winning in combat, try getting better at the game.",
        "Loading...\nTIP: Remember, players die if they are killed.",
        "Loading...\nTIP: Hitting your enemy does more damage than not hitting them.",
        "Loading...\nTIP: The best strategy for defeating enemies is to reduce their health to zero while maintaining yours above zero.",
        "Loading...\nTIP: Your enemies are against you, so be alert all times.",
        "Loading...\nTIP: The firearms work best with bullets in them.",
        "Loading...\nTIP: Staying alive increases your survival rate.",
        "Loading...\nTIP: Remember, shooting without silencer is illegal on other planets."
    };

    public void LoadRegressionScene()
    {
        StartCoroutine(LoadScene("Regression"));
        
    }
    public void LoadClassificationScene()
    {
        StartCoroutine(LoadScene("Classification"));
    }
    public void QuitApplication()
    {
        Application.Quit();
    }

    IEnumerator LoadScene(string whichScene)
    {
        mainMenuCanvas.SetActive(false);
        loadingCanvas.SetActive(true);

        AudioSource asx = GetComponent<AudioSource>();
        float maxed_volume = asx.volume;

        text.text = Functions.RandomIn(loadingTips);
        yield return new WaitForSeconds(0.33f);
        float freq = Time.deltaTime;

        while(elapsed_time < load_time)
        {
            yield return new WaitForSeconds(freq);
            elapsed_time += freq;
            loadSlide.value = elapsed_time / load_time;
            asx.volume = Mathf.Lerp(asx.volume, 0.02f, elapsed_time/load_time);
        }

        /*// wait 0.33 more second
        text.gameObject.SetActive(false);
        loadSlide.gameObject.SetActive(false);
        elapsed_time = 0f;
        while(elapsed_time < 0.33f)
        {
            yield return new WaitForSeconds(freq);
            elapsed_time += freq;
        }*/
        SceneManager.LoadScene(whichScene);
    }
}
