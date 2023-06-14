using Unity.XR.PXR;
using UnityEngine;

namespace MyScripts
{
    public class PVREyeTracking : IEyeTrackingDeviceInteract
    {
        public void Bind()
        {
            PXR_Enterprise.InitEnterpriseService("ShowConcentration");
            PXR_Enterprise.BindEnterpriseService();
        }

        public void UnBind() => 
            PXR_Enterprise.UnBindEnterpriseService();

        public void Init() => 
            PXR_Enterprise.StartVrSettingsItem(StartVRSettingsEnum.START_VR_SETTINGS_ITEM_GENERAL, true, 0);

        public bool GetEyeTrackingData(out Vector3 lGuide, out Vector3 rGuide, out Vector3 cGuide, out float lOpenness,
            out float rOpenness)
        {
            lGuide = default;
            rGuide = default;
            cGuide = default;
            lOpenness = 0;
            rOpenness = 0;
            
            bool detected = PXR_EyeTracking.GetLeftEyePositionGuide(out lGuide);
            detected = detected && PXR_EyeTracking.GetRightEyePositionGuide(out rGuide);
            detected = detected && PXR_EyeTracking.GetFoveatedGazeDirection(out cGuide);
            detected = detected && PXR_EyeTracking.GetLeftEyeGazeOpenness(out lOpenness);
            detected = detected && PXR_EyeTracking.GetRightEyeGazeOpenness(out rOpenness);

            return detected;
        }
    }
}