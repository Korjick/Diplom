// using System;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using TMPro;
using Unity.XR.PXR;
using UnityEngine;
using UnityEngine.Networking;

namespace MyScripts
{
    public class ShowConcentration : MonoBehaviour
    {
        [SerializeField] private TMP_Text _text;
        [SerializeField] private Camera _camera;

        // private readonly string _url = $"http://192.168.154.217:8000/predict";
        private readonly string _url = $"http://0.0.0.0:8000/predict";

        private Coroutine _resultCoroutine;
        private IEyeTrackingDeviceInteract _eyeTrackingDeviceInteract;

        private RaycastHit[] _raycastHits;
        
        private void Awake()
        {
            _eyeTrackingDeviceInteract = new PVREyeTracking();
            _eyeTrackingDeviceInteract.Bind();
        }

        private void Start()
        {
            _raycastHits = new RaycastHit[3];
            _eyeTrackingDeviceInteract.Init();
        }

        private void Update()
        {
            int count = Physics.RaycastNonAlloc(
                _camera.transform.position,
                _camera.transform.forward,
                _raycastHits,
                100);

            int foundedIdx = Array.FindIndex(_raycastHits, hit => hit.transform.CompareTag("Attention"));
            if (count > 0 && foundedIdx >= 0 && _resultCoroutine == null) 
                _resultCoroutine = StartCoroutine(PostRequestEverySecond());
        }

        private void OnDestroy()
        {
            _eyeTrackingDeviceInteract.UnBind();
        }

        private IEnumerator PostRequestEverySecond()
        {
            if (_eyeTrackingDeviceInteract.GetEyeTrackingData(out Vector3 lGuide, out Vector3 rGuide, out Vector3 cGuide,
                    out float lOpenness, out float rOpenness))
            {
                // Создаём тестовые данные
                InputData testData = new()
                {
                    data = new List<float>
                    {
                        cGuide.x, cGuide.y,
                        lGuide.x, lGuide.y, lGuide.z,
                        rGuide.x, rGuide.y, rGuide.z,
                        Vector2.Distance(lGuide, cGuide) * lOpenness * 100,
                        Vector2.Distance(rGuide, cGuide) * rOpenness * 100
                    }
                };
                string json = JsonUtility.ToJson(testData);

                Debug.Log($"Send To: {_url}\nValue: {json}");

                // Создаём запрос
                var request = new UnityWebRequest(_url, "POST");
                byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");

                // Отправляем запрос и ждём ответа
                yield return request.SendWebRequest();

                string text = "Concentration Index: ";
                if (request.result == UnityWebRequest.Result.Success)
                {
                    ResultData results = JsonUtility.FromJson<ResultData>(request.downloadHandler.text);

                    Debug.Log("Predictions: " + string.Join(",", results.predictions));

                    bool neutral = results.predictions.TrueForAll(p => p < 5f);
                    float maxV = results.predictions.Max();
                    int maxIdx = results.predictions.IndexOf(maxV);

                    maxV /= 10;
                    float ci = 0;

                    if (neutral)
                    {
                        ci = (maxV * 0.9f) * 100;
                    }
                    else
                    {
                        switch (maxIdx)
                        {
                            // Anger
                            case 0:
                                ci = (maxV * 0.25f) * 100;
                                break;
                            // Tenderness
                            case 1:
                                ci = (maxV * 0.6f) * 100;
                                break;
                            // Disgust
                            case 2:
                                ci = (maxV * 0.2f) * 100;
                                break;
                            // Sadness
                            case 3:
                                ci = (maxV * 0.3f) * 100;
                                break;
                        }
                    }

                    text += ci.ToString(CultureInfo.InvariantCulture) + "%";
                }
                else
                {
                    text += request.error;
                }

                _text.text = text;

                // Ждём секунду перед следующим запросом
                yield return new WaitForSeconds(1f);
            }
            
            _resultCoroutine = null;
        }
        
        //

        private class InputData
        {
            public List<float> data;
        }

        private class ResultData
        {
            public List<float> predictions;
        }
    }
}
