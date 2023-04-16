using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ExtraSettingsButton : MonoBehaviour
{
    public Sprite state1;
    public Sprite state2;
    public Button button;
    public Image image;
    public GameObject obj_to_change_activ;
    public void OnStateChanged()
    {
        // if state 1
        if(image.sprite == state1)
        {
            image.sprite = state2;
            obj_to_change_activ.SetActive(obj_to_change_activ.activeSelf == true? false:true);

        }
        // if state 2
        else
        {
            image.sprite = state1;
            obj_to_change_activ.SetActive(obj_to_change_activ.activeSelf == true ? false : true);
        }
    }


}
