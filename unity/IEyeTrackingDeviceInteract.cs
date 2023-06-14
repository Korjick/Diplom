using UnityEngine;

namespace MyScripts
{
    public interface IEyeTrackingDeviceInteract
    {
        void Bind();
        void UnBind();
        void Init();
        bool GetEyeTrackingData(out Vector3 lGuide, out Vector3 rGuide, out Vector3 cGuide, out float lOpenness,
            out float rOpenness);
    }
}