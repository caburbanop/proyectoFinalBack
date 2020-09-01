<?php

namespace App\Controller;

use Symfony\Bundle\FrameworkBundle\Controller\AbstractController;
use Symfony\Component\Routing\Annotation\Route;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\HttpFoundation\Response;
use Symfony\Component\HttpFoundation\JsonResponse;

class DefaultController extends AbstractController
{
    /**
     * @Route("/default", name="default")
     */
    public function index(Request $request) {
        $data = array(
            'estado' => 'Exito',
            'mensaje' => 'Video procesado correctamente en el servidor.',
            'proceso' => ''
        );
        try {
            $json = $request->request->get('json');
            $datosEnvio = json_decode($json);
            $nombreVideo = $datosEnvio->video[0];
            $data['proceso'] = "850";
            //print('Llego al método');
        }catch (\Exception $e) {
            $data['estado'] = "Error";
            $data['mensaje'] = "".$e->getMessage();
        }
        return new JsonResponse($data);
    }

    /**
     * @Route("/upload", name="upload")
     */
    public function uploadAction(Request $request){
        $data = array(
            'estado' => 'Exito',
            'mensaje' => 'Archivo cargado correctamente.'
        );
        try {
            $file = $request->files->get("adjuntos"); 
            $ext = strtolower($file->getClientOriginalExtension());
            $carpeta = "/tmp/upload/";
            $nombre = uniqid().".".$ext;
            $file->move($carpeta, $nombre);
            $data['nombre'] = $nombre;
        } catch (\Exception $e) {
            $data['estado'] = "Error";
            $data['mensaje'] = "".$e->getMessage();
        }
        return new JsonResponse($data);
    }

    /**
     * @Route("/consultarInfo", name="consultarInfo")
     */
    public function consultarInfo(Request $request) {
        $data = array(
            'estado' => 'Exito',
            'mensaje' => 'Información retornada.'
        );
        try {
            $json = $request->request->get('json');
            $datosEnvio = json_decode($json);
            $tipo = $datosEnvio->tipo;
            //print('Llego al método');
        }catch (\Exception $e) {
            $data['estado'] = "Error";
            $data['mensaje'] = "".$e->getMessage();
        }
        return new JsonResponse($data);
    }
}
