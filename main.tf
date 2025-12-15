terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Configure the AWS Provider
provider "aws" {
  #region = "us-east-1"
  region = "eu-west-2"
}

// ===========================================================================

resource "aws_key_pair" "ldlc_ssh_key" {
  key_name   = "ldlc_ssh_key"
  public_key = file("~/.ssh/p4greta.pub")
}

resource "aws_security_group" "allow_ssh_and_outbound" {
  name        = "allow_ssh"
  description = "Allow SSH inbound traffic"
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # À restreindre en production !
  }
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # À restreindre en production !
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "t3_micro" {
  ami="ami-046c2381f11878233" # eu-west-2 ubuntu
  instance_type = "t3.micro"
  key_name      = aws_key_pair.ldlc_ssh_key.key_name
  vpc_security_group_ids = [aws_security_group.allow_ssh_and_outbound.id]

  root_block_device {
    volume_size = 32 # Taille du disque en Go
    volume_type = "gp3" # Type de volume (gp3 est recommandé pour un bon rapport performance/prix)
  }

  tags = {
    Name = "Instance-T3-Micro"
  }
}

output "instance_public_dns" {
  value = [aws_instance.t3_micro.public_dns, aws_instance.t3_micro.public_ip]
}
