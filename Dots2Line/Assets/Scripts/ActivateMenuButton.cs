using UnityEngine;

public class ActivateMenuButton : MonoBehaviour
{
    [SerializeField] GameObject menu;
    public void ChangeMenuState() => menu.SetActive(!menu.gameObject.activeInHierarchy);
}
