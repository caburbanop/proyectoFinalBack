<?php

namespace App\Controller;

use Symfony\Bundle\FrameworkBundle\Controller\AbstractController;
use Symfony\Component\Routing\Annotation\Route;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\HttpFoundation\Response;
use Symfony\Component\HttpFoundation\JsonResponse;

use Symfony\Component\Process\Process;
use Symfony\Component\Process\Exception\ProcessFailedException;

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
            $nombreVideo = "/tmp/upload/".$datosEnvio->video[0];
            // Cambiamos de directorio
            if (chdir("/Aplicaciones/projectFinal/public/reconocimiento_grafica")) {
                $comando = "python reconocimiento.py ".$nombreVideo.' > /dev/null 2>&1 & echo $!';
                exec($comando, $output, $return_var);
                $pid = (int)$output[0];

                // Ejecutarlo por proceso
                //$comando = "python reconocimiento.py ".$nombreVideo;
                //$process = new Process(exec($comando, $output, $return_var));
                //$process = new Process($comando);
                //$process->start();
                //$pid = $process->getPid();

                $data['proceso'] = $pid;
            }
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
            'mensaje' => 'Información retornada.',
            'datos' => ''
        );
        try {
            $json = $request->request->get('json');
            $datosEnvio = json_decode($json);
            $tipo = $datosEnvio->tipo;
            $proceso = $datosEnvio->proceso;

            // Obtenemos el arreglo generado
            if (chdir("/Aplicaciones/projectFinal/public/reconocimiento_grafica")) {
                $file = "emociones.txt";
                $fp = fopen($file, "r");
                $contenido = fread($fp, filesize($file));
                $data['datos'] = $contenido;
                /*while (!feof($fp)) {
                    $linea = fgets($fp);
                } */               
            }            

            //$comando = "ps -p ".$proceso." -o comm=";
            //$result = exec($comando);
            //0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
            //print('Llego al método');
        }catch (\Exception $e) {
            $data['estado'] = "Error";
            $data['mensaje'] = "".$e->getMessage();
        }
        return new JsonResponse($data);
    }

    function isRunning($pid){
        try{
            $result = shell_exec(sprintf("ps %d", $pid));
            if( count(preg_split("/\n/", $result)) > 2){
                return true;
            }
        }catch(\Exception $e){

        }
        return false;
    }
}
