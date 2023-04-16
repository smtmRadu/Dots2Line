using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class StateChangingButton : MonoBehaviour
{
    public List<Sprite> dotSprites = new List<Sprite>();
    public int type = 0;

    public Image img;

    public void OnClick()
    {
        for (int i = 0; i < dotSprites.Count; i++)
        {
            if (img.sprite != dotSprites[i])
                continue;

            if (i == dotSprites.Count - 1)
            {
                img.sprite = dotSprites[0];
                type = 0;
            }
            else
            {
                img.sprite = dotSprites[i + 1];
                type = i + 1;
            }

            
            return;
        }
    }
}
