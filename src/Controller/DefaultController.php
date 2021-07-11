<?php

namespace App\Controller;

use Symfony\Bundle\FrameworkBundle\Controller\AbstractController;
use Symfony\Component\Routing\Annotation\Route;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\HttpFoundation\Response;
use Symfony\Component\HttpFoundation\JsonResponse;

use Symfony\Component\Process\Process;
use Symfony\Component\Process\Exception\ProcessFailedException;
use Symfony\Component\Validator\Constraints\Length;

class DefaultController extends AbstractController
{
    /**
     * @Route("/default", name="default")
     */
    public function index(Request $request) {
        $data = array(
            'estado' => 'Exito',
            'mensaje' => 'Video procesado correctamente en el servidor.',
            'proceso' => '',
            'tipo' => ''
        );
        try {
            $json = $request->request->get('json');
            $datosEnvio = json_decode($json);
            $nombreVideo = "/tmp/upload/".$datosEnvio->video[0];
            $tipo = $datosEnvio->tipo;
            $data['tipo'] = $tipo;
            if ($tipo == "facial") {
                $data['proceso'] = $this->reconocimientoFacial($nombreVideo);
            } elseif ($tipo == "habla") {
                $data['proceso'] = $this->reconocimientoHabla($nombreVideo);
            } elseif ($tipo == "facial-habla") {
                $data['proceso'] = $this->reconocimientoFacial($nombreVideo);
                $data['proceso'] = $this->reconocimientoHabla($nombreVideo);
            }
        }catch (\Exception $e) {
            $data['estado'] = "Error";
            $data['mensaje'] = "".$e->getMessage();
        }
        return new JsonResponse($data);
    }

    function reconocimientoFacial($nombreVideo){
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

            return $pid;
        }
    }

    function reconocimientoHabla($nombreVideo){
        // Cambiamos de directorio
        if (chdir("/Aplicaciones/projectFinal/public/OpenVokaturi-3-4/examples")) {
            $comando = "python extraer_audio.py ".$nombreVideo;
            exec($comando, $output, $return_var);
            $ficheros = scandir("audios-generados");
            //sort($ficheros);
            //$dir = opendir("audios-generados");
            unlink("emociones.txt");
            $identificador = 0;
            foreach ($ficheros as $fichero) {
                if( $fichero != "." && $fichero != ".."){
                    //$dividir = explode("-", $fichero);
                    //$nombre = explode(".", $dividir[1]);
                    $comando = "python OpenVokaWavMean-linux64.py audios-generados/".$fichero.' '.$identificador.' > /dev/null 2>&1 & echo $!';
                    $identificador = $identificador + 5;
                    exec($comando, $output1, $return_var);                            
                }
            }
            $pid = (int)$output1[0];
            return $pid;
            //$data['proceso'] = 0;
        }
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
            'datos' => '',
            'tipo' => ''
        );
        try {
            $json = $request->request->get('json');
            $datosEnvio = json_decode($json);
            $tipo = $datosEnvio->tipo;
            $proceso = $datosEnvio->proceso;
            $data['tipo'] = $tipo;

            // Obtenemos el arreglo generado
            if ($tipo == "facial") {                
                $data['datos'] = $this->extraerFacial();
            } elseif ($tipo == "habla") {
                $data['datos'] = $this->extraerHabla();
            } elseif ($tipo == "facial-habla") {
                // Extraemos información de facial
                $contenidoFacial = $this->extraerFacial();
                // Extraemos información de habla
                $contenidoHabla = $this->extraerHabla();

                // Validamos la información 
                $arregloFacial = explode("\n", $contenidoFacial);
                //unset($arregloFacial[count($arregloFacial)-1]);
                $arregloHabla = explode("\n", $contenidoHabla);
                //unset($arregloHabla[0]);
                $contenidoFinal = "";

                for ($i=0; $i<count($arregloFacial); $i++){
                    if ($arregloFacial[$i] != "" && $arregloHabla[$i+1] != "") {
                        $valuesFacial = explode(",", $arregloFacial[$i]);
                        $copyValuesFacial = $valuesFacial;
                        $valuesHabla = explode(",", $arregloHabla[$i+1]);
                        $copyValuesHabla = $valuesHabla;
                        unset($copyValuesFacial[0]);
                        unset($copyValuesHabla[0]);
                        $maximoFacial = max($copyValuesFacial);
                        $maximoHabla = max($copyValuesHabla);

                        // Validamos el valor mayor
                        if ($maximoFacial > $maximoHabla){
                            $contenidoFinal = $contenidoFinal.$arregloFacial[$i]."\n";
                        } else {
                            $contenidoFinal = $contenidoFinal.$arregloHabla[$i+1]."\n";
                        }
                    }                    
                }
                $data['datos'] = $contenidoFinal;
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

    function extraerFacial(){
        if (chdir("/Aplicaciones/projectFinal/public/reconocimiento_grafica")) {
            $file = "emociones.txt";
            $fp = fopen($file, "r");
            $contenido = fread($fp, filesize($file));
            return $contenido;
            /*while (!feof($fp)) {
                $linea = fgets($fp);
            } */               
        }
    }

    function extraerHabla(){
        // Cambiamos de directorio
        if (chdir("/Aplicaciones/projectFinal/public/OpenVokaturi-3-4/examples")) {
            $file = "emociones.txt";
            $fp = fopen($file, "r");
            $contenido = fread($fp, filesize($file));
            $datosArchivo = explode("\n", $contenido);
            sort($datosArchivo, 1);
            $contenido = implode("\n", $datosArchivo);
            return $contenido;
        }
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
