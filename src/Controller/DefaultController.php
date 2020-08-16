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
            'mensaje' => 'Video procesado correctamente correctamente.',
            'nombre' => ''
        );
        try {
            $json = $request->request->get('json');
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
            'mensaje' => 'Archivo cargado correctamente.',
            'nombre' => ''
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
}
